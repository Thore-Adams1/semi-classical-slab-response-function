import shutil
import os
import subprocess
from pathlib import Path
CLI_DOC_TEMPLATE_RST = """\
``{script_name}``
===================================

.. argparse::
   :module: {module_path}
   :func: get_parser
   :prog: {script_name}
   :markdownhelp:"""

def main():
    docs_dir = Path(__file__).parent
    api_doc_dir = docs_dir / "_apidocs"
    if api_doc_dir.exists():
        shutil.rmtree(api_doc_dir)
    repo_root = docs_dir.parent
    module_dir = repo_root / "scsr"
    cli_module_dir = module_dir / "cli"
    apidoc_cmd = [
        "sphinx-apidoc", "-o", api_doc_dir, module_dir, cli_module_dir,
        "-H", "SCSR API"
    ]
    return_code = subprocess.call(apidoc_cmd)
    if return_code != 0:
        print("Api doc building failed")
        exit(return_code)
    cli_doc_dir = docs_dir / "_clidocs"
    if cli_doc_dir.exists():
        shutil.rmtree(cli_doc_dir)
    cli_doc_dir.mkdir()
    for cli_script_path in sorted(cli_module_dir.glob("*.py")):
        if cli_script_path.name.startswith("_"):
            continue

        script_docs_path = cli_doc_dir / cli_script_path.with_suffix(".rst").name
        with open(script_docs_path, "w+", encoding="utf-8") as doc_file:
            script_no_suffix = cli_script_path.with_suffix("")
            module_path = str(
                script_no_suffix.relative_to(repo_root)
            ).replace(os.sep, ".")
            doc_contents = CLI_DOC_TEMPLATE_RST.format(
                script_name=f"scsr-{script_no_suffix.name.replace('_', '-')}",
                module_path=module_path
            )
            doc_file.write(doc_contents)
        print(f"Wrote {cli_script_path}")

if __name__ == "__main__":
    main()