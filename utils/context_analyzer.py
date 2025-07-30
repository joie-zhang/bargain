#!/usr/bin/env python3
"""
Context Analyzer for AI Safety Research
Intelligently selects relevant files and context for planning and implementation
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FileInfo:
    """Information about a file in the codebase"""

    path: Path
    size: int
    imports: List[str]
    exports: List[str]
    complexity_score: float
    relevance_score: float = 0.0


@dataclass
class ContextBudget:
    """Budget constraints for context selection"""

    max_files: int = 20
    max_total_size: int = 100_000  # bytes
    max_depth: int = 3  # for dependency traversal


class ContextAnalyzer:
    """Analyzes codebase and selects relevant context"""

    def __init__(self, project_root: Path = Path.current()):
        self.project_root = project_root
        self.file_cache: Dict[Path, FileInfo] = {}
        self.dependency_graph: Dict[Path, Set[Path]] = defaultdict(set)

    def analyze_request(self, request: str) -> Dict[str, List[str]]:
        """Extract key terms and concepts from user request"""
        # Simple keyword extraction - in production, use NLP
        keywords = set()

        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', request)
        keywords.update(quoted)

        # Extract camelCase and snake_case identifiers
        identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", request)
        keywords.update(identifiers)

        # Common action words that suggest file types
        action_keywords = {
            "api": ["routes", "endpoints", "controllers", "views"],
            "test": ["test_", "_test", "spec", "tests/"],
            "model": ["models", "schemas", "entities"],
            "config": ["config", "settings", "env"],
            "cache": ["cache", "redis", "memcache"],
            "database": ["db", "models", "migrations"],
        }

        suggested_patterns = []
        for word in keywords:
            word_lower = word.lower()
            for category, patterns in action_keywords.items():
                if category in word_lower:
                    suggested_patterns.extend(patterns)

        return {"keywords": list(keywords), "patterns": suggested_patterns}

    def find_relevant_files(
        self, keywords: List[str], patterns: List[str], budget: ContextBudget
    ) -> List[FileInfo]:
        """Find files relevant to the keywords and patterns"""
        relevant_files = []

        # Get all files in the project
        all_files = self._get_project_files()

        # Score each file based on relevance
        for file_path in all_files:
            if len(relevant_files) >= budget.max_files:
                break

            file_info = self._analyze_file(file_path)
            if not file_info:
                continue

            # Calculate relevance score
            score = 0.0
            path_str = str(file_path).lower()
            content = self._read_file_preview(file_path)

            # Check keywords in path and content
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in path_str:
                    score += 2.0  # Path match is strong signal
                if keyword_lower in content.lower():
                    score += 1.0

            # Check patterns
            for pattern in patterns:
                if pattern.lower() in path_str:
                    score += 1.5

            file_info.relevance_score = score

            if score > 0:
                relevant_files.append(file_info)

        # Sort by relevance and return top files within budget
        relevant_files.sort(key=lambda f: f.relevance_score, reverse=True)

        # Apply size budget
        total_size = 0
        filtered_files = []
        for file in relevant_files:
            if total_size + file.size <= budget.max_total_size:
                filtered_files.append(file)
                total_size += file.size
            else:
                break

        return filtered_files

    def expand_context(
        self, core_files: List[FileInfo], budget: ContextBudget
    ) -> List[FileInfo]:
        """Expand context to include dependencies and related files"""
        expanded = set(f.path for f in core_files)
        to_process = list(expanded)
        depth = 0

        while to_process and depth < budget.max_depth:
            next_batch = []

            for file_path in to_process:
                # Find imports/dependencies
                deps = self._find_dependencies(file_path)

                for dep in deps:
                    if dep not in expanded and len(expanded) < budget.max_files:
                        expanded.add(dep)
                        next_batch.append(dep)

            to_process = next_batch
            depth += 1

        # Convert back to FileInfo objects
        result = []
        for path in expanded:
            file_info = self._analyze_file(path)
            if file_info:
                result.append(file_info)

        return result

    def generate_context_summary(self, files: List[FileInfo]) -> str:
        """Generate a summary of the selected context"""
        summary = ["# Context Analysis Summary\n"]

        # Group files by directory
        by_dir = defaultdict(list)
        for file in files:
            dir_path = file.path.parent
            by_dir[dir_path].append(file)

        summary.append("## Selected Files by Directory\n")
        for dir_path, dir_files in sorted(by_dir.items()):
            summary.append(f"\n### {dir_path}/")
            for file in sorted(dir_files, key=lambda f: f.path.name):
                size_kb = file.size / 1024
                summary.append(
                    f"- {file.path.name} ({size_kb:.1f}KB) - relevance: {file.relevance_score:.1f}"
                )

        # Add statistics
        total_size = sum(f.size for f in files)
        summary.append(f"\n## Statistics")
        summary.append(f"- Total files: {len(files)}")
        summary.append(f"- Total size: {total_size / 1024:.1f}KB")
        summary.append(
            f"- Average relevance: {sum(f.relevance_score for f in files) / len(files):.1f}"
        )

        return "\n".join(summary)

    def _get_project_files(self) -> List[Path]:
        """Get all project files, respecting .gitignore"""
        try:
            # Use git ls-files if available
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            files = [
                self.project_root / f for f in result.stdout.strip().split("\n") if f
            ]
            return [f for f in files if f.is_file()]
        except subprocess.CalledProcessError:
            # Fallback to walking directory
            files = []
            exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}

            for root, dirs, filenames in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for filename in filenames:
                    if not filename.startswith("."):
                        files.append(Path(root) / filename)

            return files

    def _analyze_file(self, file_path: Path) -> Optional[FileInfo]:
        """Analyze a single file"""
        if file_path in self.file_cache:
            return self.file_cache[file_path]

        try:
            size = file_path.stat().st_size

            # Skip large files
            if size > 1_000_000:  # 1MB
                return None

            # Extract imports/exports based on file type
            imports = []
            exports = []

            if file_path.suffix == ".py":
                content = file_path.read_text()
                imports = re.findall(r"^(?:from|import)\s+(\S+)", content, re.MULTILINE)
                exports = re.findall(r"^(?:class|def)\s+(\w+)", content, re.MULTILINE)
            elif file_path.suffix in [".js", ".ts"]:
                content = file_path.read_text()
                imports = re.findall(
                    r'(?:import|require)\s*\(?["\']([^"\']+)["\']\)?', content
                )
                exports = re.findall(
                    r"export\s+(?:class|function|const)\s+(\w+)", content
                )

            # Simple complexity score based on file size and structure
            complexity = (size / 1000) + len(imports) * 0.1 + len(exports) * 0.2

            file_info = FileInfo(
                path=file_path,
                size=size,
                imports=imports,
                exports=exports,
                complexity_score=complexity,
            )

            self.file_cache[file_path] = file_info
            return file_info

        except Exception:
            return None

    def _read_file_preview(self, file_path: Path, max_lines: int = 100) -> str:
        """Read first N lines of a file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                return "".join(lines)
        except Exception:
            return ""

    def _find_dependencies(self, file_path: Path) -> List[Path]:
        """Find dependencies of a file"""
        if file_path in self.dependency_graph:
            return list(self.dependency_graph[file_path])

        deps = set()
        file_info = self._analyze_file(file_path)

        if not file_info:
            return []

        # Resolve imports to actual files
        for import_str in file_info.imports:
            # Handle relative imports
            if import_str.startswith("."):
                base_dir = file_path.parent
                parts = import_str.split(".")

                # Go up directories for each leading dot
                for _ in range(len(parts) - len(parts.lstrip("."))):
                    base_dir = base_dir.parent

                # Try to find the imported module
                module_path = base_dir / parts[-1].replace(".", "/")

                # Check various extensions
                for ext in [".py", ".js", ".ts", "/index.py", "/index.js", "/index.ts"]:
                    potential_path = Path(str(module_path) + ext)
                    if potential_path.is_file():
                        deps.add(potential_path)
                        break

        self.dependency_graph[file_path] = deps
        return list(deps)


def main():
    """CLI interface for context analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze context for implementation planning"
    )
    parser.add_argument("request", help="Description of what you want to implement")
    parser.add_argument(
        "--max-files", type=int, default=20, help="Maximum files to include"
    )
    parser.add_argument(
        "--max-size", type=int, default=100000, help="Maximum total size in bytes"
    )
    parser.add_argument(
        "--expand", action="store_true", help="Expand to include dependencies"
    )

    args = parser.parse_args()

    analyzer = ContextAnalyzer()

    # Analyze the request
    print("Analyzing request...")
    analysis = analyzer.analyze_request(args.request)
    print(f"Keywords: {', '.join(analysis['keywords'])}")
    print(f"Patterns: {', '.join(analysis['patterns'])}")

    # Find relevant files
    budget = ContextBudget(max_files=args.max_files, max_total_size=args.max_size)
    print(
        f"\nSearching for relevant files (budget: {budget.max_files} files, {budget.max_total_size / 1024:.0f}KB)..."
    )

    relevant_files = analyzer.find_relevant_files(
        analysis["keywords"], analysis["patterns"], budget
    )

    if args.expand and relevant_files:
        print("Expanding context to include dependencies...")
        relevant_files = analyzer.expand_context(relevant_files, budget)

    # Generate summary
    if relevant_files:
        summary = analyzer.generate_context_summary(relevant_files)
        print(summary)

        print("\n## Suggested command:")
        file_args = " ".join(f"@{f.path}" for f in relevant_files[:5])
        print(f"claude")
        print(f"> {file_args} {args.request}")
    else:
        print("\nNo relevant files found. Try more specific keywords.")


if __name__ == "__main__":
    main()
