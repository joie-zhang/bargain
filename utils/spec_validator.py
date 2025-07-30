#!/usr/bin/env python3
"""
Specification Validator for AI Safety Research
Validates that specifications meet quality standards and can generate tests
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of specification validation"""

    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, any]


@dataclass
class Requirement:
    """A single requirement from the specification"""

    id: str
    type: str  # MUST, SHOULD, MAY
    description: str
    testable: bool


@dataclass
class AcceptanceCriteria:
    """A single acceptance criterion"""

    id: str
    given: str
    when: str
    then: str
    tested: bool = False


class SpecificationValidator:
    """Validates and analyzes specification documents"""

    def __init__(self):
        self.required_sections = [
            "Overview",
            "Why",
            "What",
            "How",
            "Testing Strategy",
            "Definition of Done",
        ]

    def validate_specification(self, spec_path: Path) -> ValidationResult:
        """
        Validate a specification file

        Args:
            spec_path: Path to specification markdown file

        Returns:
            ValidationResult with errors, warnings, and metrics
        """
        errors = []
        warnings = []
        metrics = {}

        if not spec_path.exists():
            return ValidationResult(False, ["Specification file not found"], [], {})

        with open(spec_path, "r") as f:
            content = f.read()

        # Check required sections
        missing_sections = self._check_required_sections(content)
        errors.extend([f"Missing required section: {s}" for s in missing_sections])

        # Extract and validate components
        requirements = self._extract_requirements(content)
        acceptance_criteria = self._extract_acceptance_criteria(content)
        test_cases = self._extract_test_cases(content)

        # Validate requirements
        req_errors, req_warnings = self._validate_requirements(requirements)
        errors.extend(req_errors)
        warnings.extend(req_warnings)

        # Validate acceptance criteria
        ac_errors, ac_warnings = self._validate_acceptance_criteria(acceptance_criteria)
        errors.extend(ac_errors)
        warnings.extend(ac_warnings)

        # Check test coverage
        coverage_warnings = self._check_test_coverage(
            requirements, acceptance_criteria, test_cases
        )
        warnings.extend(coverage_warnings)

        # Calculate metrics
        metrics = {
            "requirement_count": len(requirements),
            "acceptance_criteria_count": len(acceptance_criteria),
            "test_case_count": len(test_cases),
            "testable_requirements": sum(1 for r in requirements if r.testable),
            "must_requirements": sum(1 for r in requirements if r.type == "MUST"),
            "should_requirements": sum(1 for r in requirements if r.type == "SHOULD"),
            "may_requirements": sum(1 for r in requirements if r.type == "MAY"),
        }

        # Calculate test coverage percentage
        if requirements:
            metrics["requirement_test_coverage"] = (
                metrics["testable_requirements"] / len(requirements) * 100
            )
        else:
            metrics["requirement_test_coverage"] = 0

        valid = len(errors) == 0

        return ValidationResult(valid, errors, warnings, metrics)

    def _check_required_sections(self, content: str) -> List[str]:
        """Check for required sections in the specification"""
        missing = []
        for section in self.required_sections:
            if f"## {section}" not in content and f"# {section}" not in content:
                missing.append(section)
        return missing

    def _extract_requirements(self, content: str) -> List[Requirement]:
        """Extract requirements from the specification"""
        requirements = []

        # Pattern for requirements like **[REQ-001]** The system MUST...
        req_pattern = (
            r"\*\*\[REQ-(\d+)\]\*\*\s+The system (MUST|SHOULD|MAY)\s+(.+?)(?=\n|$)"
        )

        for match in re.finditer(req_pattern, content, re.MULTILINE):
            req_id = f"REQ-{match.group(1)}"
            req_type = match.group(2)
            description = match.group(3).strip()

            # Simple heuristic for testability
            testable = any(
                keyword in description.lower()
                for keyword in [
                    "return",
                    "output",
                    "generate",
                    "calculate",
                    "validate",
                    "process",
                    "convert",
                    "create",
                    "update",
                    "delete",
                ]
            )

            requirements.append(Requirement(req_id, req_type, description, testable))

        return requirements

    def _extract_acceptance_criteria(self, content: str) -> List[AcceptanceCriteria]:
        """Extract acceptance criteria from the specification"""
        criteria = []

        # Pattern for Given/When/Then format
        ac_pattern = (
            r"\*\*AC-(\d+)\*\*:\s*Given\s+(.+?),\s*when\s+(.+?),\s*then\s+(.+?)(?=\n|$)"
        )

        for match in re.finditer(ac_pattern, content, re.IGNORECASE | re.MULTILINE):
            ac_id = f"AC-{match.group(1)}"
            given = match.group(2).strip()
            when = match.group(3).strip()
            then = match.group(4).strip()

            criteria.append(AcceptanceCriteria(ac_id, given, when, then))

        return criteria

    def _extract_test_cases(self, content: str) -> List[str]:
        """Extract test case IDs from the specification"""
        test_cases = []

        # Pattern for test cases like **[TEST-001]**
        test_pattern = r"\*\*\[TEST-(\d+)\]\*\*"

        for match in re.finditer(test_pattern, content):
            test_cases.append(f"TEST-{match.group(1)}")

        return test_cases

    def _validate_requirements(
        self, requirements: List[Requirement]
    ) -> Tuple[List[str], List[str]]:
        """Validate requirements for consistency and quality"""
        errors = []
        warnings = []

        # Check for duplicate IDs
        req_ids = [r.id for r in requirements]
        duplicates = set([x for x in req_ids if req_ids.count(x) > 1])
        if duplicates:
            errors.append(f"Duplicate requirement IDs: {', '.join(duplicates)}")

        # Check for untestable MUST requirements
        untestable_must = [
            r for r in requirements if r.type == "MUST" and not r.testable
        ]
        if untestable_must:
            warnings.append(
                f"MUST requirements that may be hard to test: {', '.join([r.id for r in untestable_must])}"
            )

        # Check for vague requirements
        vague_keywords = ["appropriate", "adequate", "suitable", "proper", "reasonable"]
        for req in requirements:
            if any(keyword in req.description.lower() for keyword in vague_keywords):
                warnings.append(
                    f"{req.id} contains vague language: {req.description[:50]}..."
                )

        return errors, warnings

    def _validate_acceptance_criteria(
        self, criteria: List[AcceptanceCriteria]
    ) -> Tuple[List[str], List[str]]:
        """Validate acceptance criteria for completeness"""
        errors = []
        warnings = []

        # Check for incomplete criteria
        for ac in criteria:
            if not ac.given or not ac.when or not ac.then:
                errors.append(f"{ac.id} is incomplete")

            # Check for measurable outcomes
            if ac.then and not any(
                keyword in ac.then.lower()
                for keyword in [
                    "should",
                    "must",
                    "will",
                    "returns",
                    "displays",
                    "shows",
                ]
            ):
                warnings.append(f"{ac.id} may not have a measurable outcome")

        return errors, warnings

    def _check_test_coverage(
        self,
        requirements: List[Requirement],
        criteria: List[AcceptanceCriteria],
        test_cases: List[str],
    ) -> List[str]:
        """Check if all requirements and criteria have corresponding tests"""
        warnings = []

        # Check if each requirement has at least one test
        for req in requirements:
            if req.testable and not any(req.id in test for test in test_cases):
                warnings.append(f"{req.id} has no corresponding test case")

        # Check if each acceptance criterion has a test
        for ac in criteria:
            if not any(ac.id in test for test in test_cases):
                warnings.append(f"{ac.id} has no corresponding test case")

        return warnings

    def generate_test_skeleton(self, spec_path: Path) -> str:
        """
        Generate a test skeleton based on the specification

        Args:
            spec_path: Path to specification file

        Returns:
            Python test skeleton as string
        """
        with open(spec_path, "r") as f:
            content = f.read()

        requirements = self._extract_requirements(content)
        criteria = self._extract_acceptance_criteria(content)

        # Extract feature name from the specification
        feature_match = re.search(r"# Specification:\s*(.+)", content)
        feature_name = feature_match.group(1) if feature_match else "Feature"

        test_code = f'''#!/usr/bin/env python3
"""
Generated test skeleton for: {feature_name}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This skeleton provides test cases for all requirements and acceptance criteria
defined in the specification. Implement each test to validate the specification.
"""

import pytest
from typing import Any

class Test{feature_name.replace(" ", "")}:
    """Test suite for {feature_name}"""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        # Initialize test data and dependencies
        test_data = {{
            "input": "sample_input",
            "expected": "expected_output",
            "config": {{"test_mode": True}}
        }}
        return test_data
'''

        # Generate tests for requirements
        if requirements:
            test_code += "\n    # Requirement Tests\n"
            for req in requirements:
                if req.testable:
                    test_code += f'''
    def test_{req.id.lower().replace("-", "_")}(self, setup):
        """
        Requirement: {req.type} {req.description}
        """
        # Test implementation for {req.id}
        # Verify that: {req.description}
        test_data = setup
        
        # Example implementation - replace with actual test logic
        result = self._process_requirement(test_data["input"])
        assert result is not None, f"{req.id} validation failed"
        
    def _process_requirement(self, input_data):
        """Helper method for requirement processing"""
        # Actual implementation would go here
        return input_data
'''

        # Generate tests for acceptance criteria
        if criteria:
            test_code += "\n    # Acceptance Criteria Tests\n"
            for ac in criteria:
                test_code += f'''
    def test_{ac.id.lower().replace("-", "_")}(self, setup):
        """
        Acceptance Criteria: {ac.id}
        Given: {ac.given}
        When: {ac.when}
        Then: {ac.then}
        """
        # Test implementation for {ac.id}
        
        # Given: {ac.given}
        initial_state = self._setup_initial_state(setup)
        
        # When: {ac.when}
        result = self._perform_action(initial_state)
        
        # Then: {ac.then}
        assert self._verify_outcome(result), f"{ac.id} validation failed"
        
    def _setup_initial_state(self, test_data):
        """Set up the initial state for testing"""
        return {{"state": "initialized", "data": test_data}}
    
    def _perform_action(self, state):
        """Perform the action being tested"""
        return {{"result": "success", "state": state}}
    
    def _verify_outcome(self, result):
        """Verify the expected outcome"""
        return result.get("result") == "success"
'''

        test_code += '''
    # Edge Case Tests
    def test_edge_case_empty_input(self, setup):
        """Test handling of empty input"""
        result = self._process_requirement("")
        assert result == "", "Empty input should return empty result"

    def test_edge_case_large_input(self, setup):
        """Test handling of large input"""
        large_input = "x" * 10000
        result = self._process_requirement(large_input)
        assert len(result) <= 10000, "Should handle large inputs gracefully"

    # Integration Tests
    def test_integration_with_validator(self, setup):
        """Test integration with specification validator"""
        from spec_validator import SpecificationValidator
        validator = SpecificationValidator()
        assert validator is not None, "Validator should be importable"
'''

        return test_code


def main():
    """CLI interface for specification validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and analyze specifications")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a specification")
    validate_parser.add_argument("spec", help="Path to specification file")

    # Generate tests command
    generate_parser = subparsers.add_parser(
        "generate-tests", help="Generate test skeleton"
    )
    generate_parser.add_argument("spec", help="Path to specification file")
    generate_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    validator = SpecificationValidator()

    if args.command == "validate":
        result = validator.validate_specification(Path(args.spec))

        print(f"Specification: {args.spec}")
        print(f"Valid: {'✅ Yes' if result.valid else '❌ No'}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  ❌ {error}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

        if result.metrics:
            print("\nMetrics:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")

    elif args.command == "generate-tests":
        test_code = validator.generate_test_skeleton(Path(args.spec))

        if args.output:
            with open(args.output, "w") as f:
                f.write(test_code)
            print(f"Generated test skeleton: {args.output}")
        else:
            print(test_code)


if __name__ == "__main__":
    main()
