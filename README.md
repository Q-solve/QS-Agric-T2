# QS-Agric-T2 = TEAM QULIMA
QULIMA - Quantum-Hybrid Maize Yield Forecasting System

PROJECT PURPOSE

Qulima (derived from "Quantum" + "Kilimo" - the Swahili word for agriculture) is a quantum-hybrid machine learning system designed to forecast maize yields in Kenyan counties by June 30th of each growing season. This project addresses SDG Target 2.0 (Zero hunger) by providing early-season yield forecasting to support food security planning, agricultural insurance, and resource allocation decisions. 

OUR CORE PROBLEM STATEMENT

Agricultural stakeholders (Government & Farmers) in Kenya lack reliable, early-season yield forecasts to inform planting decisions, insurance pricing, and food security interventions. Traditional forecasting methods often rely on data unavailable until harvest time, creating a critical information gap during the growing season.

OUR SOLUTION

Qulima combines classical machine learning baselines (random forest, Cat boost, Xgboost) with quantum-inspired feature mapping and quantum kernel regression to:

-Predict maize yield (tonnes/hectare) using only data available by June 30th 2026.

-Provide county-level forecasts for Trans Nzoia and Uasin Gishu.

-Establish reproducible, leakage-free pipelines for agricultural forecasting.

## Repository Structure
The repository is organized as follows:

raw data/ – Raw and intermediate maize statistics generated during internal processing
docs/ – Documentation related to the dataset
notebooks/ – Data generated during data processing and analysis.
Quantum/ - The quantum modelling obtained results.

Key Innovation

The system uses quantum kernel methods to capture complex nonlinear relationships in agricultural data that classical models may miss, while maintaining strict chronological validation and data integrity standards.


Prerequisites;

-Google colab (Classical modelling and data cleaning).

-Qbraid account (for quantum simulations).

INSTALLATION GUIDE;

Clone the repository

bash
   git clone <repo-url>
   cd qsolve-2026-team2-maize-yield
   
Set up a Python environment using Google Colab (This one needs to be reviewed).
bash
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
Set up qBraid (for the quantum/hybrid model)
Create or log in to your qBraid account.
Open the repository's notebooks/ folder inside the qBraid environment, or install the qBraid SDK locally per the qBraid platform documentation.

Verify the setup
bash
   python -m pytest tests/          # if a tests/ folder exists
   jupyter notebook notebooks/00_setup_check.ipynb
   
Usage
Download and audit raw data — run the scripts in src/data_download/ to pull maize production/harvested-area records and CHIRPS rainfall, then log the download in docs/provenance_log.md.

Build the modelling table — run src/preprocessing/build_county_year_table.py (or the equivalent notebook) to join sources into the documented county-year table in data/processed/, applying only pre-30-June features.

Run the classical baseline — notebooks/01_classical_baseline.ipynb fits the historical mean/trend baseline and the competitive classical model (e.g. random forest), using the chronological train/validation/test split.

Run the quantum/hybrid model — notebooks/02_quantum_model.ipynb runs the quantum-kernel or variational regression model on the qBraid simulator, using the same split and feature set.

Compare results — notebooks/03_evaluation.ipynb produces the classical-vs-quantum results table, the actual-vs-predicted plot, and county-level error breakdown, saved to results/trans_nzoia_uasin_gishu/.

Read the write-up — docs/limitations.md documents assumptions, uncertainty, and the scalability/adoption discussion required for the final presentation.
Contact
    Role	         Name	           Roles
1. Lead mentor -Kisilu Wambua. (Provides guidance on problem understanding, method selection, and technical review)
2. Data Lead-	Avery Inyangala. (manages sourcing, cleaning, joins, provenance and reproducibility.)	
3. Classical modelling lead	-Bruce Kinyanjui (builds and validates the baseline)	
4. Quantum lead - Mona Tanei. (owns the formulation, implementation and resource reporting.)
5. Evaluation & presentation lead - Adika Awino. ( coordinates comparison, visualisation, documentation and
the final demonstration.)
6. Problem and agriculture lead-Kennedy Mutugi.	( keeps the work grounded in users, local constraints and SDG 2
outcomes.)

Team coordination:post updates, questions, and blockers in the team's assigned Google Classroom rather than only in chat, so there's a record for the contribution log.

Issues with this repository: open a GitHub Issue in this repo and tag the relevant lead from the table above.
Competition-level questions (scoring, official dataset releases, regional allocation changes): route through the lead mentor to the organising committee.

Additional resources
https://github.com/HarvestStat/HarvestStat-Africa/tree/main

