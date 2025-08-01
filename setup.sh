#!/bin/bash

# Trap to handle cleanup on exit
trap exit_gracefully SIGINT SIGTERM EXIT
revert_on_exit=true

# Functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

execute_command() {
    echo -e "  \033[36m$ $1\033[0m"
    eval "$1"
}

prompt_yes_no() {
    echo -e "\n\033[1;34m? $1\033[0m"
    while true; do
        if ! read -r -p "  $2 [Y/n]: " response; then
            # Handle EOF (Ctrl+D)
            echo -e ""
            exit 1
        fi
        case $response in
            [Yy]* | '') eval "$3='y'"; return 0 ;;
            [Nn]*) eval "$3='n'"; return 0 ;;
            *) echo "  Please answer yes (y) or no (n)." ;;
        esac
    done
}

exit_gracefully() {
    if [ "$revert_on_exit" = true ]; then
        echo -e "\n\033[31mAborting Setup:\033[0m"

        # Revert to initial commit
        echo -e "  Reverting changes to initial commit:"
        execute_command "git reset --hard $(git rev-list --max-parents=0 HEAD)"
        execute_command "git clean -fd"

        # Remove .pixi directory if it exists
        if [ -d ".pixi" ]; then
            echo -e "  Removing .pixi directory:"
            execute_command "rm -rf .pixi"
        fi

        revert_on_exit=false
        exit 1
    fi
}

# ? Setup Mode
prompt_yes_no "Setup Mode" "Wanna sit back and enjoy the ride (accept all defaults)?" use_opinionated_setup

if [ "${use_opinionated_setup}" = "y" ]; then
    echo -e "\n  \033[32m✔ Using opinionated setup with recommended options.\033[0m"

    run_pixi="y"
    install_hooks="y"
    create_readme="y"
else
    echo -e "\n\033[33mℹ You will be prompted for each option during the setup.\033[0m"

    run_pixi=""
    install_hooks=""
    create_readme=""
fi

# == Package Name ==

echo -e "\n\033[1m== Package Name ==\033[0m"

# ? Package Name
current_name="package_name"
name=$(basename "$(pwd)" | tr '-' '_')

find . -type f -not -path '*/\.*' -not -name 'setup.sh' -exec sh -c '
    if file -b --mime-type "$1" | grep -q "^text/"; then
        if [ "$(uname)" = "Darwin" ]; then
            # macOS (BSD sed)
            sed -i "" "s/$2/$3/g" "$1"
        else
            # Linux (GNU sed)
            sed -i "s/$2/$3/g" "$1"
        fi
    fi
' sh {} "$current_name" "$name" \;

if [ -d "$current_name" ]; then
    mv "$current_name" "$name"
    echo -e "  \033[32m✔ Directory renamed to $name\033[0m"
fi

echo -e "  \033[32m✔ Package renamed to $name\033[0m"

# == Install Python Environment ==

echo -e "\n\033[1m== Install Python Environment ==\033[0m"
if [ -z "$run_pixi" ]; then
    prompt_yes_no "Pixi Install" "Do you want to run 'pixi install'?" run_pixi
fi

if [ "${run_pixi}" = "y" ]; then
    execute_command "pixi install"
    echo -e "  \033[32m✔ 'pixi install' completed.\033[0m"

    # == Use Pre-commit Hooks ==

    echo -e "\n\033[1m== Use Pre-commit Hooks ==\033[0m"
    if [ -z "$install_hooks" ]; then
        prompt_yes_no "Pre-commit Hooks" "Do you want to use pre-commit hooks?" install_hooks
    fi

    if [ "${install_hooks}" = "y" ]; then
        execute_command "pixi r pre-commit install"
        echo -e "  \033[32m✔ Pre-commit hooks installed successfully.\033[0m"
    else
        echo -e "\n  Removing pre-commit configuration files and dependency..."
        execute_command "rm -f .pre-commit-config.yaml"
        execute_command "rm -f .github/workflows/code-style.yaml"
        execute_command "sed -i '' '/pre-commit = \"*\"/d' pyproject.toml"
        echo -e "  \033[32m✔ Pre-commit configuration files and dependency removed.\033[0m"
    fi
fi

# == Create README ==

echo -e "\n\033[1m== Create README ==\033[0m"

if [ -z "$create_readme" ]; then
    prompt_yes_no "README" "Do you want to create a README.md skeleton?" create_readme
fi

if [ "${create_readme}" = "y" ]; then
    execute_command "rm README.md"
    execute_command "mv template_README.md README.md"
    echo -e "  \033[32m✔ Created README.md for new project (skeleton only).\033[0m"
else
    execute_command "rm README.md template_README.md"
    echo -e "  \033[33mℹ Deleted 'project-starter' README.md.\033[0m"
fi

# == Clean Up ==

echo -e "\n\033[1m== Clean Up ==\033[0m"

script_path=$(realpath "$0")
execute_command "rm \"$script_path\""
echo -e "  \033[32m🗑️ Setup script has been deleted.\033[0m"

# Final message
echo -e "\n\033[1m🎉 == Setup Complete! == 🎉\033[0m"
echo -e "\n  \033[33mNote: To undo and start over, simply run:\033[0m"
echo -e "  \033[36m  git reset --hard $(git rev-list --max-parents=0 HEAD) && git clean -fd\033[0m"

revert_on_exit=false

# Ask about committing and pushing
prompt_yes_no "Commit and Push" "Do you want to commit and push these changes?" commit_and_push

if [ "${commit_and_push}" = "y" ]; then
    echo -e "\n\033[1m== Committing and Pushing Changes ==\033[0m"
    execute_command "git add ."
    execute_command "git commit -m 'Initial setup with project-starter'"
    execute_command "git pull"
    execute_command "git push origin main"
    echo -e "  \033[32m✔ Changes committed and pushed successfully.\033[0m"
else
    echo -e "\n\033[33mℹ Changes have not been committed or pushed.\033[0m"
fi
