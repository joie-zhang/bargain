#!/bin/bash

# Script to add co-author to all local commits
COAUTHOR="Co-authored-by: Joie Zhang <107374257+joie-zhang@users.noreply.github.com>"

# Get the number of commits ahead of origin/params_updated
NUM_COMMITS=$(git rev-list --count origin/params_updated..HEAD)

if [ "$NUM_COMMITS" -eq 0 ]; then
    echo "No local commits to modify"
    exit 0
fi

echo "Adding co-author to $NUM_COMMITS local commits..."

# Use git filter-branch to rewrite commit messages
git filter-branch -f --msg-filter "
    cat
    if ! grep -q 'Co-authored-by: Joie Zhang' ; then
        echo ''
        echo '$COAUTHOR'
    fi
" origin/params_updated..HEAD

echo "Successfully added co-author to all local commits!"
echo "You can now push with: git push origin params_updated"