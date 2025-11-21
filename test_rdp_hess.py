from dataclasses import dataclass
from pathlib import Path, PurePath

import numpy as np
import sirf.STIR as STIR

import stir

def sirf_to_stir(image: STIR.ImageData) -> stir.FloatVoxelsOnCartesianGrid:
    r"""Convert sirf.STIR.ImageData to a stir object

    Warning: writes to tmp.* in current directory (could be done better)
    """
    image.write('tmp.hv')
    return stir.FloatVoxelsOnCartesianGrid.read_from_file('tmp.hv')


def stir_to_sirf(image: stir.FloatVoxelsOnCartesianGrid) -> STIR.ImageData:
    r"""Convert stir image to sirf.STIR.ImageData

    Warning: writes to tmp.* in current directory (could be done better)
    """
    image.write_to_file('tmp.hv')
    return STIR.ImageData('tmp.hv')


def construct_stir_RDP(sirf_prior: STIR.RelativeDifferencePrior, initial_image: STIR.ImageData):
    '''
    Construct a stir smoothed Relative Difference Prior (RDP), given input from a sirf.STIR one

    `initial_image` is used to determine a smoothing factor (epsilon).
    `kappa` is used to pass voxel-dependent weights.
    '''
    prior = stir.GibbsRelativeDifferencePenalty3DFloat()
    # need to make it differentiable
    prior.set_epsilon(sirf_prior.get_epsilon())
    prior.set_penalisation_factor(sirf_prior.get_penalisation_factor())
    if sirf_prior.get_kappa() is not None:
        prior.set_kappa_sptr(sirf_to_stir(sirf_prior.get_kappa()))
    prior.set_up(sirf_to_stir(initial_image))
    return prior

def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    """
    Construct a smoothed Relative Difference Prior (RDP)

    initial_image: used to determine a smoothing factor (epsilon).
    kappa: used to pass voxel-dependent weights.
    """
    prior = getattr(STIR, 'CudaRelativeDifferencePrior', STIR.RelativeDifferencePrior)()
    # need to make it differentiable
    epsilon = initial_image.max() * max_scaling
    prior.set_epsilon(epsilon)
    prior.set_penalisation_factor(penalty_strength)
    prior.set_kappa(kappa)
    prior.set_up(initial_image)
    return prior


@dataclass
class Dataset:
    acquired_data: STIR.AcquisitionData
    additive_term: STIR.AcquisitionData
    mult_factors: STIR.AcquisitionData
    OSEM_image: STIR.ImageData
    prior: STIR.RelativeDifferencePrior
    kappa: STIR.ImageData
    reference_image: STIR.ImageData | None
    whole_object_mask: STIR.ImageData | None
    background_mask: STIR.ImageData | None
    voi_masks: dict[str, STIR.ImageData]
    FOV_mask: STIR.ImageData
    path: PurePath


def get_data(srcdir, outdir, sirf_verbosity=0, read_sinos=True):
    """
    Load data from `srcdir`, constructs prior and return as a `Dataset`.
    Also redirects sirf.STIR log output to `outdir`, unless that's set to None
    """
    srcdir = Path(srcdir)
    STIR.set_verbosity(sirf_verbosity)                # set to higher value to diagnose problems
    STIR.AcquisitionData.set_storage_scheme('memory') # needed for get_subsets()

    if outdir is not None:
        outdir = Path(outdir)
        _ = STIR.MessageRedirector(str(outdir / 'info.txt'), str(outdir / 'warnings.txt'), str(outdir / 'errors.txt'))
    acquired_data = STIR.AcquisitionData(str(srcdir / 'prompts.hs')) if read_sinos else None
    additive_term = STIR.AcquisitionData(str(srcdir / 'additive_term.hs')) if read_sinos else None
    mult_factors = STIR.AcquisitionData(str(srcdir / 'mult_factors.hs')) if read_sinos else None
    OSEM_image = STIR.ImageData(str(srcdir / 'OSEM_image.hv'))
    # Find FOV mask
    # WARNING: we are currently using Parralelproj with default settings, which uses a cylindrical FOV.
    # The current code gives identical results to thresholding the sensitivity image (for those settings)
    FOV_mask = STIR.TruncateToCylinderProcessor().process(OSEM_image.allocate(1))
    kappa = STIR.ImageData(str(srcdir / 'kappa.hv'))
    if (penalty_strength_file := (srcdir / 'penalisation_factor.txt')).is_file():
        penalty_strength = float(np.loadtxt(penalty_strength_file))
    else:
        penalty_strength = 1 / 700 # default choice
    prior = construct_RDP(penalty_strength, OSEM_image, kappa)

    def get_image(fname):
        if (source := srcdir / 'PETRIC' / fname).is_file():
            return STIR.ImageData(str(source))
        return None # explicit to suppress linter warnings

    reference_image = get_image('reference_image.hv')
    whole_object_mask = get_image('VOI_whole_object.hv')
    background_mask = get_image('VOI_background.hv')
    voi_masks = {
        voi.stem[4:]: STIR.ImageData(str(voi))
        for voi in (srcdir / 'PETRIC').glob("VOI_*.hv") if voi.stem[4:] not in ('background', 'whole_object')}

    return Dataset(acquired_data, additive_term, mult_factors, OSEM_image, prior, kappa, reference_image,
                   whole_object_mask, background_mask, voi_masks, FOV_mask, srcdir.resolve())


if __name__ == "__main__":
    #srcdir  = Path("/mnt/share/petric") / "Siemens_mMR_NEMA_IQ"
    srcdir  = Path("/mnt/share/petric") / "NeuroLF_Hoffman_Dataset"
    data = get_data(srcdir=srcdir, outdir=Path("."), sirf_verbosity=1)

    sirf_prior = data.prior  
    initial = data.OSEM_image

    # construct the RDP in STIR
    stir_initial = sirf_to_stir(initial)
    stir_prior = construct_stir_RDP(sirf_prior, initial)

    # compute the diagonal of the prior Hessian
    diagH_prior_stir = stir_initial.clone()
    stir_prior.compute_Hessian_diagonal(diagH_prior_stir, stir_initial)
    diagH_prior_sirf = stir_to_sirf(diagH_prior_stir)
