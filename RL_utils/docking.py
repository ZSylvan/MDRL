"""Serial docking with task-supplied targets and a separate log per molecule."""

import math
from pathlib import Path
import re
import subprocess


def sdf_to_pdbqt(input_sdf, output_pdbqt):
    from openbabel import openbabel

    conversion = openbabel.OBConversion()
    if not conversion.SetInAndOutFormats('sdf', 'pdbqt'):
        raise RuntimeError('OpenBabel SDF/PDBQT format plugins are unavailable')
    molecule = openbabel.OBMol()
    if not conversion.ReadFile(molecule, str(input_sdf)):
        raise RuntimeError('OpenBabel could not read ' + str(input_sdf))
    molecule.AddHydrogens()
    charge_model = openbabel.OBChargeModel.FindType('gasteiger')
    if charge_model is None or not charge_model.ComputeCharges(molecule):
        raise RuntimeError('OpenBabel Gasteiger charge assignment failed')
    if not conversion.WriteFile(molecule, str(output_pdbqt)):
        raise RuntimeError('OpenBabel could not write ' + str(output_pdbqt))
    conversion.CloseOutFile()


def parse_affinity(output):
    match = re.search(r'^\s*1\s+([-+]?\d+(?:\.\d+)?)\s+', output, re.MULTILINE)
    if match is None:
        raise ValueError('Vina output contains no mode-1 affinity')
    affinity = float(match.group(1))
    if not math.isfinite(affinity):
        raise ValueError('Nonfinite docking affinity')
    return affinity


def normalize_affinity(affinity, parameters):
    mu, sigma = parameters['mu'], parameters['sigma']
    clipped = max(affinity, mu) if parameters['minimize'] else min(affinity, mu)
    return math.exp(-0.5 * ((clipped - mu) / sigma) ** 2)


def run_vina(ligand, target, settings, logfile):
    command = [settings['executable'], '--receptor', target['receptor'], '--ligand', str(ligand)]
    for axis, center, size in zip('xyz', target['center'], target['box_size']):
        command.extend(['--center_' + axis, str(center), '--size_' + axis, str(size)])
    command.extend(['--exhaustiveness', str(settings['exhaustiveness']),
                    '--num_modes', str(settings['num_modes']), '--cpu', str(settings['cpu']),
                    '--out', str(Path(logfile).with_suffix('.pdbqt'))])
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=settings.get('timeout'), check=False)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        Path(logfile).write_text(decoded(exc.stdout) + '\n' + decoded(exc.stderr) + '\nTIMEOUT\n')
        raise RuntimeError('Docking timed out for ' + target['name']) from exc
    Path(logfile).write_text(result.stdout + '\nSTDERR:\n' + result.stderr)
    if result.returncode:
        raise RuntimeError('Vina failed for %s (exit %d); see %s' % (target['name'], result.returncode, logfile))
    affinity = parse_affinity(result.stdout)
    return dict(target=target['name'], affinity=affinity,
                normalized_score=normalize_affinity(affinity, target['normalization']),
                log=str(logfile))


def dock_molecule(molecule, targets, directory, settings):
    from rdkit import Chem

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sdf = directory / 'ligand.sdf'
    ligand = directory / 'ligand.pdbqt'
    Chem.MolToMolFile(molecule, str(sdf))
    sdf_to_pdbqt(sdf, ligand)
    return [run_vina(ligand, target, settings, directory / (target['name'] + '.log'))
            for target in targets]
