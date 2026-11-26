# Contributing

When contributing to this repository, please first [create an issue](https://github.com/ComputationalPhysiology/simcardemsx/issues/new) describing the bug you would like to fix or the feature you would like to add. There you can discuss the change with the maintainers before you make it.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## New contributor guide

To get an overview of the project, read the [documentation](https://computationalphysiology.github.io/simcardemsx). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Setting up a development environment

simcardemsx needs [DOLFINx](https://github.com/FEniCS/dolfinx), which cannot be installed from PyPI on its own. Develop inside the DOLFINx container, which is also what CI tests in:

```bash
docker run -ti -v $(pwd):/home/shared -w /home/shared ghcr.io/fenics/dolfinx/dolfinx:stable
```

The repository also has a VS Code [dev container](https://code.visualstudio.com/docs/devcontainers/containers) in `.devcontainer/`, which starts from the same image and runs the steps below for you.

Inside the container, install the package in editable mode with all its extras:

```bash
python3 -m pip install --group build
HDF5_MPI=ON HDF5_PKGCONFIG_NAME="hdf5" python3 -m pip install h5py --no-build-isolation --no-binary=h5py
python3 -m pip install scifem --no-build-isolation
python3 -m pip install -e ".[all]"
```

The first three lines build two packages against the container's own libraries rather than taking wheels from PyPI:

- h5py, against the container's MPI-enabled HDF5. simcardemsx writes its results and checkpoints through it.
- scifem, which beat and pulse use, against the container's DOLFINx.

## Running the tests

```bash
python3 -m pytest                    # the whole suite, with a coverage report
python3 -m pytest -m "not slow"      # only the fast tests
python3 -m pytest tests/test_ode_model.py -v
```

CI runs the suite in both the `stable` and the `nightly` DOLFINx containers, so a change must work with both. A new feature needs tests, and a bug fix needs a test that fails without the fix.

## Documentation

The documentation is a [MyST](https://mystmd.org) site. [`docs/README.md`](docs/README.md) explains how to build it locally. Note the following:

- The API reference is generated from the docstrings, so a new public function or class needs a docstring.
- A new module needs a line in the API reference section of `docs/myst.yml`.
- The tutorials in `examples/tutorials/` are executed when the site is built, so a tutorial that raises an error fails CI. A new tutorial needs a line in the tutorials section of `docs/myst.yml`.

## Changes that belong in another package

simcardemsx couples solvers from other packages, maintained by the same people:

- [fenicsx-beat](https://github.com/finsberg/fenicsx-beat) for electrophysiology;
- [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) for mechanics and circulation;
- [gotranx](https://github.com/finsberg/gotranx) for the code generated from `.ode` files;
- [crossbridge](https://github.com/ComputationalPhysiology/crossbridge) for the NumPy contraction models.

If a bug or a missing feature is in one of them, please fix it there, with an issue or a pull request against that repository, rather than working around it in simcardemsx.

## Pull Request Process

### Pull Request

- When you're finished with the changes, create a pull request, also known as a PR. It is also OK to create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) from the very beginning. Once you are done you can click on the "Ready for review" button. You can also [request a review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) from one of the maintainers.
- Describe what the PR changes and why, and don't forget to [link the PR to the issue that you opened](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a team member will review your proposal. We may ask questions or request for additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, see GitHub's guide to [resolving a merge conflict](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).
- Please make sure that all tests are passing, that the documentation builds, and that code coverage is not lower than before your contribution. You can see the different GitHub Actions workflows by clicking the "Actions" tab in the GitHub repository.

### Enforced style guide using pre-commit hooks

We want to have a consistent style on all the contributions to the repository. The way we enforce this is through pre-commit hooks, and contributors are encouraged to install the pre-commit hooks locally when developing. `pre-commit` is part of the `dev` extra, so the install above already has it. Install the hooks using the command
```
pre-commit install
```
at the root of the repository. This will install all the hooks listed in the file called `.pre-commit-config.yaml` in the root of the repository:

- [ruff](https://docs.astral.sh/ruff/) lints and formats the code, with a line length of 100;
- [mypy](https://mypy-lang.org) checks the types. It runs in your own environment, so the package and its dependencies must be installed there;
- [cspell](https://cspell.org) checks the spelling. Add a domain-specific word to `.cspell_dict.txt` rather than rewording around it.

Every time you make a commit to the repository a set of tests will run to make sure that the changes you made are following the style guide. Usually, the hooks will autoformat your code so that you only need to do a `git add` again and then redo the `git commit`. You can also run the pre-commit hooks on all the files using the command
```
pre-commit run --all
```
To learn more about pre-commit you can check out https://pre-commit.com


## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at henriknf@simula.no. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
