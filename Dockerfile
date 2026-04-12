FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    openmm=8.1 \
    pdbfixer \
    mdtraj \
    ambertools \
    pytraj \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH

COPY . /app
RUN mkdir -p inputs out && chmod +x run.sh
