#!/usr/bin/env python3
"""
Codebase Integration Script
Helps integrate external codebases into the research workflow.
"""

import argparse
import subprocess
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil


class CodebaseIntegrator:
    def __init__(self):
        self.integration_dir = Path("external_codebases")
        self.integration_dir.mkdir(exist_ok=True)

    def clone_or_download(self, source: str, target_name: Optional[str] = None) -> Path:
        """Clone or download a codebase from various sources"""
        if source.startswith(("http://", "https://")) and source.endswith(".git"):
            # Git repository
            repo_name = target_name or source.split("/")[-1].replace(".git", "")
            target_path = self.integration_dir / repo_name

            if target_path.exists():
                print(f"✅ Repository already exists at: {target_path}")
                # Pull latest changes
                subprocess.run(["git", "pull"], cwd=target_path)
            else:
                print(f"📥 Cloning repository: {source}")
                subprocess.run(["git", "clone", source, str(target_path)])

            return target_path
        elif source.startswith(("http://", "https://")):
            # Direct download
            print(f"📥 Downloading: {source}")
            # Implementation for downloading archives
            pass
        else:
            # Local path
            source_path = Path(source)
            if source_path.exists():
                target_path = self.integration_dir / (target_name or source_path.name)
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, target_path)
                return target_path
            else:
                raise ValueError(f"Source not found: {source}")

    def analyze_codebase(self, codebase_path: Path) -> Dict[str, Any]:
        """Analyze the structure and contents of a codebase"""
        print(f"🔍 Analyzing codebase: {codebase_path}")

        analysis = {
            "path": str(codebase_path),
            "name": codebase_path.name,
            "structure": {},
            "languages": {},
            "key_files": [],
            "dependencies": {},
        }

        # Count files by extension
        for root, dirs, files in os.walk(codebase_path):
            # Skip hidden and build directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "build", "dist"]
            ]

            for file in files:
                if file.startswith("."):
                    continue

                ext = Path(file).suffix.lower()
                if ext:
                    analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1

                # Identify key files
                if file in [
                    "README.md",
                    "setup.py",
                    "package.json",
                    "requirements.txt",
                    "Cargo.toml",
                ]:
                    analysis["key_files"].append(os.path.join(root, file))

        # Identify project type and dependencies
        if (codebase_path / "package.json").exists():
            with open(codebase_path / "package.json", "r") as f:
                pkg = json.load(f)
                analysis["project_type"] = "nodejs"
                analysis["dependencies"]["npm"] = list(
                    pkg.get("dependencies", {}).keys()
                )

        if (codebase_path / "requirements.txt").exists():
            with open(codebase_path / "requirements.txt", "r") as f:
                analysis["project_type"] = "python"
                analysis["dependencies"]["pip"] = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

        return analysis

    def create_integration_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create integration configuration based on analysis"""
        config = {
            "codebase": {
                "name": analysis["name"],
                "path": analysis["path"],
                "type": analysis.get("project_type", "unknown"),
            },
            "integration": {"commands": [], "hooks": [], "workflows": []},
        }

        # Add language-specific commands
        if ".py" in analysis["languages"]:
            config["integration"]["commands"].extend(
                [
                    {
                        "name": "analyze_python_functions",
                        "command": "python scripts/commands/analyze_function.py",
                        "args": "--codebase {path}",
                    }
                ]
            )

        # Add project-specific workflows
        if analysis.get("project_type") == "python":
            config["integration"]["workflows"].append(
                {
                    "name": "python_safety_analysis",
                    "steps": [
                        {
                            "tool": "multi_mind",
                            "args": {"focus": "security vulnerabilities"},
                        },
                        {"tool": "analyze_function", "args": {"focus": "complexity"}},
                    ],
                }
            )

        return config

    def generate_claude_md(
        self, analysis: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate CLAUDE.md content for the integrated codebase"""
        claude_md = f"""# {analysis["name"]} Integration

## Codebase Overview
- **Path**: `{analysis["path"]}`
- **Type**: {analysis.get("project_type", "Unknown")}
- **Primary Languages**: {", ".join(sorted(analysis["languages"].keys())[:5])}

## Key Files
{chr(10).join(f"- {file}" for file in analysis["key_files"][:10])}

## Integration Commands

### Analysis Commands
```bash
# Analyze functions
python scripts/commands/analyze_function.py <function_name> --codebase {analysis["path"]}

# Multi-mind analysis
python scripts/commands/multi_mind.py "Analyze security of {analysis["name"]}" --output ./analysis

# Generate specifications
python scripts/commands/spec_driven.py "<feature>" --output ./specs
```

### Orchestrated Workflows
```bash
# Comprehensive analysis
python scripts/commands/orchestrate_research.py \\
    --workflow comprehensive_analysis \\
    --codebase {analysis["path"]} \\
    --task "Analyze {analysis["name"]} architecture and security"

# Custom workflow
python scripts/commands/orchestrate_research.py \\
    --custom-steps custom_workflow.json \\
    --codebase {analysis["path"]}
```

## Quick Tasks

1. **Security Analysis**
   ```bash
   /multi-mind "Analyze security vulnerabilities in {analysis["name"]}"
   ```

2. **Architecture Review**
   ```bash
   gemini -p "@{analysis["path"]}" "Explain the architecture and design patterns"
   ```

3. **Code Quality Assessment**
   ```bash
   /analyze-function main --codebase {analysis["path"]}
   ```

## Notes
- This codebase was integrated on {Path(analysis["path"]).stat().st_mtime}
- Use the orchestration tools to coordinate complex analyses
- Always verify findings with multiple tools
"""
        return claude_md

    def save_integration(self, analysis: Dict[str, Any], config: Dict[str, Any]):
        """Save integration configuration and documentation"""
        integration_name = analysis["name"]
        integration_path = Path(".claude/integrations") / integration_name
        integration_path.mkdir(parents=True, exist_ok=True)

        # Save analysis
        with open(integration_path / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Save config
        with open(integration_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save CLAUDE.md
        claude_md = self.generate_claude_md(analysis, config)
        with open(integration_path / "CLAUDE.md", "w") as f:
            f.write(claude_md)

        print(f"✅ Integration saved to: {integration_path}")
        print(f"📄 CLAUDE.md generated for easy reference")

    def integrate(self, source: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Main integration workflow"""
        # Download/clone codebase
        codebase_path = self.clone_or_download(source, name)

        # Analyze structure
        analysis = self.analyze_codebase(codebase_path)

        # Create integration config
        config = self.create_integration_config(analysis)

        # Save integration
        self.save_integration(analysis, config)

        return {"path": codebase_path, "analysis": analysis, "config": config}


def main():
    parser = argparse.ArgumentParser(description="Integrate External Codebase")
    parser.add_argument("source", help="GitHub URL, local path, or archive URL")
    parser.add_argument("--name", type=str, help="Custom name for the integration")
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze, don't save"
    )

    args = parser.parse_args()

    integrator = CodebaseIntegrator()

    try:
        result = integrator.integrate(args.source, args.name)

        print("\n" + "=" * 80)
        print("INTEGRATION COMPLETE")
        print("=" * 80)
        print(f"Codebase: {result['analysis']['name']}")
        print(f"Location: {result['path']}")
        print(
            f"Languages: {', '.join(list(result['analysis']['languages'].keys())[:5])}"
        )
        print(f"\nNext steps:")
        print(f"1. Review: .claude/integrations/{result['analysis']['name']}/CLAUDE.md")
        print(
            f"2. Run: python scripts/commands/orchestrate_research.py --codebase {result['path']}"
        )

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return 1


if __name__ == "__main__":
    main()
