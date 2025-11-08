# HPP-BioHealth

## Install requirements

```bash
pip install -r requirements.txt
```

rename `.env_example` to  `.env` file with your OpenAI and K2-Think API keys.

## Run

```bash
python gradio_app.py
```

and it will start a gradio app at `http://0.0.0.0:7860`

## default example

We’ve provided an example located in `example/case1`, and the OCR-processed `diagnosis.json` file has already been
included, so you can proceed directly with the analysis without performing OCR.

## Structure of the repo

This part of the data is derived from the HPP dataset provided by MBZUAI and is intended to describe the causal chains of different diseases.

### hpp_date

`node.json`: Contains all node information across the four major systems.
`edge.json`: Contains the causal relationships between different indicators in the HPP dataset.
