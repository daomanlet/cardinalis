Project Description



Folder Structure Conventions
============================

> Folder structure options and naming conventions for software projects

### A typical top-level directory layout

    .
    ├── models                  # The tools for training CV models
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── app                     # Web Service backend 
    ├── binoculars              # Frontend to shows bird species
    ├── samples                  # End points samples, AI model samples etc
    ├── deploy                  # Docker files for deploying 
    ├── LICENSE
    └── README.md

> Use short lowercase names at least for the top-level files and folders except
> `LICENSE`, `README.md`