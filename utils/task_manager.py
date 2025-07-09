#!/usr/bin/env python3
"""
Task Manager for AI Safety Research
Manages tasks with strict separation of why/what/how, inspired by Backlog.md
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in-progress"
    DONE = "done"


class TaskPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task:
    """Represents a single task with structured metadata"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = self._load_content()
        self.metadata = self._parse_metadata()
        self.acceptance_criteria = self._parse_acceptance_criteria()
        self.implementation_steps = self._parse_implementation_steps()

    def _load_content(self) -> str:
        """Load task content from file"""
        with open(self.file_path, "r") as f:
            return f.read()

    def _parse_metadata(self) -> Dict:
        """Parse task metadata from content"""
        metadata = {
            "id": self._extract_field("ID"),
            "status": self._extract_field("Status"),
            "priority": self._extract_field("Priority"),
            "created": self._extract_field("Created"),
            "updated": self._extract_field("Updated"),
            "assigned": self._extract_field("Assigned"),
            "parent_task": self._extract_field("Parent Task"),
            "related_issues": self._extract_field("Related Issues"),
            "specification": self._extract_field("Specification"),
            "title": self._extract_title(),
        }
        return metadata

    def _extract_field(self, field_name: str) -> Optional[str]:
        """Extract a field value from the metadata section"""
        pattern = rf"- \*\*{field_name}\*\*:\s*(.+?)(?=\n|$)"
        match = re.search(pattern, self.content)
        return match.group(1).strip() if match else None

    def _extract_title(self) -> str:
        """Extract task title from the first heading"""
        match = re.search(r"^# Task:\s*(.+?)(?=\n|$)", self.content, re.MULTILINE)
        return match.group(1) if match else "Untitled Task"

    def _parse_acceptance_criteria(self) -> List[Dict]:
        """Parse acceptance criteria checkboxes"""
        criteria = []

        # Find the What section
        what_section = re.search(r"## What.*?\n(.*?)(?=## |$)", self.content, re.DOTALL)

        if what_section:
            content = what_section.group(1)
            # Find all checkbox items
            checkbox_pattern = r"- \[([ x])\]\s+(.+?)(?=\n|$)"
            for match in re.finditer(checkbox_pattern, content, re.MULTILINE):
                checked = match.group(1) == "x"
                description = match.group(2).strip()
                criteria.append({"checked": checked, "description": description})

        return criteria

    def _parse_implementation_steps(self) -> List[Dict]:
        """Parse implementation plan steps"""
        steps = []

        # Find the How section
        how_section = re.search(r"## How.*?\n(.*?)(?=## |$)", self.content, re.DOTALL)

        if how_section:
            content = how_section.group(1)
            # Parse nested checkboxes
            lines = content.split("\n")
            current_section = None

            for line in lines:
                # Main section
                main_match = re.match(r"^\d+\.\s*\[([ x])\]\s*\*\*(.+?)\*\*", line)
                if main_match:
                    current_section = {
                        "checked": main_match.group(1) == "x",
                        "title": main_match.group(2),
                        "substeps": [],
                    }
                    steps.append(current_section)

                # Substep
                elif current_section:
                    substep_match = re.match(r"^\s+-\s*\[([ x])\]\s*(.+)", line)
                    if substep_match:
                        current_section["substeps"].append(
                            {
                                "checked": substep_match.group(1) == "x",
                                "description": substep_match.group(2).strip(),
                            }
                        )

        return steps

    def get_status(self) -> TaskStatus:
        """Get task status as enum"""
        status_str = self.metadata.get("status", "todo").strip("`")
        try:
            return TaskStatus(status_str)
        except ValueError:
            return TaskStatus.TODO

    def get_priority(self) -> TaskPriority:
        """Get task priority as enum"""
        priority_str = self.metadata.get("priority", "medium").strip("`")
        try:
            return TaskPriority(priority_str)
        except ValueError:
            return TaskPriority.MEDIUM

    def update_status(self, new_status: TaskStatus) -> None:
        """Update task status in file"""
        old_status_pattern = r"(- \*\*Status\*\*:\s*)`[^`]+`"
        new_status_str = f"\\1`{new_status.value}`"
        self.content = re.sub(old_status_pattern, new_status_str, self.content)

        # Update the Updated field
        now = datetime.now().strftime("%Y-%m-%d")
        self.content = re.sub(r"(- \*\*Updated\*\*:\s*).+", f"\\1{now}", self.content)

        # Save back to file
        with open(self.file_path, "w") as f:
            f.write(self.content)

    def check_criterion(self, criterion_index: int) -> None:
        """Check off an acceptance criterion"""
        # This is simplified - in production would need more robust parsing
        count = 0
        lines = self.content.split("\n")

        for i, line in enumerate(lines):
            if re.match(r"- \[ \]", line):
                if count == criterion_index:
                    lines[i] = line.replace("- [ ]", "- [x]")
                    break
                count += 1

        self.content = "\n".join(lines)
        with open(self.file_path, "w") as f:
            f.write(self.content)


class TaskManager:
    """Manages collection of tasks"""

    def __init__(self, tasks_dir: str = "tasks"):
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(exist_ok=True)

    def create_task(
        self,
        title: str,
        why: str,
        what: List[str],
        how: List[str],
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> Path:
        """Create a new task from components"""
        # Generate task ID
        date_str = datetime.now().strftime("%Y-%m-%d")
        existing_tasks = list(self.tasks_dir.glob(f"task-{date_str}-*.md"))
        next_num = len(existing_tasks) + 1
        task_id = f"TASK-{date_str}-{next_num:03d}"

        # Create filename
        safe_title = re.sub(r"[^\w\s-]", "", title.lower())
        safe_title = re.sub(r"[-\s]+", "-", safe_title)
        filename = f"task-{date_str}-{next_num:03d}-{safe_title}.md"
        file_path = self.tasks_dir / filename

        # Generate content
        content = f"""# Task: {title}

## Meta
- **ID**: {task_id}
- **Status**: `todo`
- **Priority**: `{priority.value}`
- **Created**: {date_str}
- **Updated**: {date_str}
- **Assigned**: [Unassigned]
- **Parent Task**: None
- **Related Issues**: []
- **Specification**: None

## Why (Purpose)
{why}

## What (Acceptance Criteria)
Clear, testable criteria that define when this task is complete:

"""
        # Add acceptance criteria
        for criterion in what:
            content += f"- [ ] {criterion}\n"

        content += """
## How (Implementation Plan)
Step-by-step approach to complete this task:

"""
        # Add implementation steps
        for i, step in enumerate(how, 1):
            content += f"{i}. [ ] **{step}**\n"

        content += """
## Technical Context
### Dependencies
- None identified yet

### Files to Modify
- TBD

### Commands
```bash
# Build command
# Test command
# Lint command
```

## Progress Notes
### {date_str} - Created
- Task created

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Tests written and passing
- [ ] No linting errors
- [ ] Documentation updated
- [ ] Task status updated to `done`

## Lessons Learned
[To be filled after completion]
""".replace("{date_str}", date_str)

        # Write file
        with open(file_path, "w") as f:
            f.write(content)

        return file_path

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List all tasks, optionally filtered by status"""
        tasks = []

        for task_file in self.tasks_dir.glob("task-*.md"):
            task = Task(task_file)
            if status is None or task.get_status() == status:
                tasks.append(task)

        # Sort by priority and creation date
        priority_order = {
            TaskPriority.HIGH: 0,
            TaskPriority.MEDIUM: 1,
            TaskPriority.LOW: 2,
        }
        tasks.sort(
            key=lambda t: (
                priority_order[t.get_priority()],
                t.metadata.get("created", ""),
            )
        )

        return tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID"""
        for task_file in self.tasks_dir.glob("task-*.md"):
            task = Task(task_file)
            if task.metadata.get("id") == task_id:
                return task
        return None

    def generate_board_view(self) -> str:
        """Generate a Kanban-style board view"""
        todo_tasks = self.list_tasks(TaskStatus.TODO)
        in_progress_tasks = self.list_tasks(TaskStatus.IN_PROGRESS)
        done_tasks = self.list_tasks(TaskStatus.DONE)

        board = "# Task Board\n\n"
        board += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        # Todo column
        board += "## 📋 To Do\n"
        for task in todo_tasks:
            priority_emoji = {
                TaskPriority.HIGH: "🔴",
                TaskPriority.MEDIUM: "🟡",
                TaskPriority.LOW: "🟢",
            }[task.get_priority()]
            board += f"- {priority_emoji} **{task.metadata['id']}**: {task.metadata['title']}\n"

        # In Progress column
        board += "\n## 🚧 In Progress\n"
        for task in in_progress_tasks:
            completed = sum(1 for c in task.acceptance_criteria if c["checked"])
            total = len(task.acceptance_criteria)
            board += f"- **{task.metadata['id']}**: {task.metadata['title']} ({completed}/{total})\n"

        # Done column
        board += "\n## ✅ Done\n"
        for task in done_tasks[:10]:  # Show only recent 10
            board += f"- **{task.metadata['id']}**: {task.metadata['title']}\n"

        return board


def main():
    """CLI interface for task management"""
    import argparse

    parser = argparse.ArgumentParser(description="Manage research tasks")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create task command
    create_parser = subparsers.add_parser("create", help="Create a new task")
    create_parser.add_argument("title", help="Task title")
    create_parser.add_argument("--why", required=True, help="Why this task is needed")
    create_parser.add_argument(
        "--what", nargs="+", required=True, help="Acceptance criteria"
    )
    create_parser.add_argument(
        "--how", nargs="+", required=True, help="Implementation steps"
    )
    create_parser.add_argument(
        "--priority", choices=["high", "medium", "low"], default="medium"
    )

    # List tasks command
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=["todo", "in-progress", "done"])

    # Update status command
    update_parser = subparsers.add_parser("update", help="Update task status")
    update_parser.add_argument("task_id", help="Task ID")
    update_parser.add_argument("status", choices=["todo", "in-progress", "done"])

    # Board view command
    board_parser = subparsers.add_parser("board", help="Show Kanban board")

    args = parser.parse_args()

    manager = TaskManager()

    if args.command == "create":
        priority = TaskPriority(args.priority)
        file_path = manager.create_task(
            args.title, args.why, args.what, args.how, priority
        )
        print(f"Created task: {file_path}")

    elif args.command == "list":
        status = TaskStatus(args.status) if args.status else None
        tasks = manager.list_tasks(status)

        for task in tasks:
            status_emoji = {
                TaskStatus.TODO: "📋",
                TaskStatus.IN_PROGRESS: "🚧",
                TaskStatus.DONE: "✅",
            }[task.get_status()]

            print(f"{status_emoji} {task.metadata['id']}: {task.metadata['title']}")
            print(f"   Priority: {task.get_priority().value}")
            print(f"   Created: {task.metadata['created']}")
            print()

    elif args.command == "update":
        task = manager.get_task(args.task_id)
        if task:
            task.update_status(TaskStatus(args.status))
            print(f"Updated {args.task_id} to {args.status}")
        else:
            print(f"Task {args.task_id} not found")

    elif args.command == "board":
        print(manager.generate_board_view())


if __name__ == "__main__":
    main()
