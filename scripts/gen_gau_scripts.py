#!/usr/bin/env python3
"""gen_gau_scripts.py — generate gau_scripts/Gau_*.py from custom_models.py

Reads custom_models.py from the repository root and writes one forwarding
wrapper per registered model, identical to the existing Gau_*.py files except for
the method name, so that you do not have to hand-write boilerplate.

Recognized registrations:
  - declarative: ASE_CALCULATORS_EXTRA = {'mymodel': dict(...), ...}
  - functional:  @register_method('mymodel')

Usage:
    python scripts/gen_gau_scripts.py            # write the missing wrappers
    python scripts/gen_gau_scripts.py --force    # overwrite existing wrappers
    python scripts/gen_gau_scripts.py --dry-run  # print what would be written
"""

import os
import re
import sys


WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
"""Gaussian External interface for {name} (start the MLIP server first)."""
from _relay import relay

if __name__ == '__main__':
    relay('{name}')
'''


def collect_custom_model_names(root):
    """Collect model names from the source of custom_models.py without running it."""
    path = os.path.join(root, 'custom_models.py')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()

    names = []
    # 1) keys of the ASE_CALCULATORS_EXTRA dictionary
    m = re.search(r'ASE_CALCULATORS_EXTRA\s*=\s*\{', src)
    if m:
        for km in re.finditer(
                r"^\s*['\"]([A-Za-z0-9_]+)['\"]\s*:", src[m.end():], re.M):
            names.append(km.group(1))
    # 2) @register_method('name') decorators
    names += re.findall(
        r"@register_method\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)", src)

    # Deduplicate while preserving order.
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def main():
    force = '--force' in sys.argv
    dry_run = '--dry-run' in sys.argv or '--dry' in sys.argv

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = collect_custom_model_names(root)
    if names is None:
        print('No custom_models.py found. Copy scripts/custom_models.example.py to '
              'custom_models.py in the repository root and fill it in first.')
        sys.exit(1)
    if not names:
        print('custom_models.py registers no models '
              '(ASE_CALCULATORS_EXTRA is empty and there are no @register_method '
              'decorators).')
        return

    gau_dir = os.path.join(root, 'gau_scripts')
    os.makedirs(gau_dir, exist_ok=True)
    created, skipped = [], []

    for name in names:
        wrapper = os.path.join(gau_dir, f'Gau_{name}.py')
        if os.path.exists(wrapper) and not force:
            skipped.append(wrapper)
            continue
        if dry_run:
            created.append(wrapper)
            continue
        with open(wrapper, 'w', encoding='utf-8') as f:
            f.write(WRAPPER_TEMPLATE.format(name=name))
        os.chmod(wrapper, 0o755)
        created.append(wrapper)

    for w in created:
        print(('[dry-run] would create: ' if dry_run else 'created: ') + w)
    if skipped:
        print(f'skipped {len(skipped)} existing wrapper(s); pass --force to overwrite')


if __name__ == '__main__':
    main()
