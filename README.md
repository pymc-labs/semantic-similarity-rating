# 🚀 PyMC Labs Project Starter

All project code goes off the rails to some extent.
It happens in tiny increments with awkward commits that "solves the problem".
This is entirely forgivable when a codebase is not designed for the new problem and there are time constraints.
It is inevitable.

However, this chaos can be mitigated with some decent guardrails. This **project template** provides the following:

- 📦 **`pixi`** for dependency and environment management.
- 🧹 **`pre-commit`** for formatting, spellcheck, etc. If everyone uses the same standard formatting, then PRs won't have flaky formatting updates that distract from the actual contribution. Reviewing code will be much easier.
- 🏷️ **`beartype`** for runtime type checking. If you know what's going in and out of functions just by reading the code, then it's easier to debug. And if these types are even enforced at runtime with tools like `beartype`, then there's a whole class of bugs that can never enter your code.
- 🧪 **`pytest`** for testing. Meanwhile, with `beartype` handling type checks, tests do not have to assert types, and can merely focus on whether the actual logic works.
- 🔄 **Github Actions** for running the pre-commit checks on each PR, automated testing and dependency management (dependabot).

## Usage

This is a pretty minimal template,
that assumes you have opinions and may want to add/remove stuff too.
To use it as intended (not that you have to),
you should put your main model logic in the `package_name/models.py` file,
adjacent logic split into sibling files,
and then have a script that imports from `package_name` and runs the model,
e.g. in a `scripts/run_model.py` file,
or a notebook somewhere (e.g. in `experimentation`).

### Prerequisites

- Python 3.11 or higher
- [Pixi package manager](https://pixi.sh/latest/)

### Get started

1. On GitHub, click on the green **Use this template** button, and create a new repository.
2. Git clone the new repository to your local machine.
3. Run the setup script `bash setup.sh` and follow the instructions.

<video src="https://github.com/user-attachments/assets/4a1ab682-bdc6-4ac9-90ad-013157c1128d" controls></video>

### Philosophy

There are some example files in this repository.
Have a look at them.
Each of the files in `package_name` has a docstring that explains their role in your package.
You may not want to follow this dogma entirely,
but having split out the code for custom types, main model logic, preprocessing, string parsing, etc.
into separate files is always a good idea.
Mostly, however, you can use these as an example to build upon.

And please contribute. If you add some guardrails that you think would be generally useful, please make a PR.
