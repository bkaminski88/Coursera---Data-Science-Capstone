# Predicting Falcon 9 First-Stage Landing Success

IBM Applied Data Science Capstone (Coursera, DS0321EN).

SpaceX advertises Falcon 9 launches at roughly $62M against $165M+ for competitors, and the
difference comes largely from reusing the first stage. So if you can predict whether the
first stage will land successfully, you can estimate the real cost of a launch — useful for
anyone bidding against SpaceX. This project builds that prediction end to end: collecting
launch data, engineering a landing-outcome label, exploring it, and comparing four
classifiers.

## Notebooks

In pipeline order:

| Notebook | What it does |
|---|---|
| `jupyter-labs-spacex-data-collection-api.ipynb` | Pulls past launches from the public SpaceX REST API v4, dereferences rocket / launchpad / payload / core IDs into flat columns (booster version, payload mass, orbit, launch site, coordinates, landing outcome, grid fins, legs, reuse counts). Filters to Falcon 9, fills missing payload mass with the column mean. |
| `jupyter-labs-webscraping.ipynb` | Scrapes a pinned Wikipedia revision of *List of Falcon 9 and Falcon Heavy launches* with BeautifulSoup, parsing the launch tables into a DataFrame as a second, independent data source. |
| `labs-jupyter-spacex-Data wrangling.ipynb` | Builds the training label. Counts launches per site and per orbit, then collapses the many `Outcome` strings into a binary `landing_class` (1 = landed, 0 = did not), which is what every later notebook predicts. |
| `jupyter-labs-eda-sql-coursera_sqllite.ipynb` | Loads the dataset into SQLite and answers ten questions in SQL — unique launch sites, total NASA (CRS) payload mass (45,596 kg), first successful ground-pad landing (2015-12-22), boosters carrying maximum payload, ranked landing outcomes over a date range. |
| `edadataviz.ipynb` | Visual EDA with matplotlib/seaborn — flight number vs. site, payload vs. orbit, success rate by orbit type, yearly success trend — then one-hot encodes the categorical features into the model-ready matrix. |
| `lab_jupyter_launch_site_location.ipynb` | Folium geospatial analysis. Maps the launch sites, clusters success/failure markers per launch, and computes great-circle distances to coastline, railway, highway and nearest city to look for siting patterns. |
| `SpaceX_Machine Learning Prediction_Part_5 V1.ipynb` | Standardizes features, splits 80/20, and tunes Logistic Regression, SVM, Decision Tree and KNN with `GridSearchCV`, comparing them by cross-validation score and confusion matrix. |

## Results

On the held-out test set (18 samples), Logistic Regression, SVM and the Decision Tree all
reach **0.8333** accuracy; KNN trails at 0.6111. The Decision Tree has the best
cross-validation score (0.8750 vs. ~0.819 for LR and SVM), so it is selected as the best
model — but the three-way tie on test accuracy is the honest summary, and with only 18 test
samples the differences are not meaningful.

The confusion matrices show the dominant error mode is false positives: predicting a
successful landing that did not happen.

> Note: the final comparison cell in the ML notebook prints an accuracy of 0.9167. That cell
> scores against the training split, not the test split, so it overstates performance. The
> test-set numbers above are the ones to read.

## Running

The notebooks were written for the course environment — some run in JupyterLite/Pyodide
(using `piplite`) rather than a local kernel. Libraries used across the set: `pandas`,
`numpy`, `requests`, `beautifulsoup4`, `matplotlib`, `seaborn`, `folium`, `scikit-learn`,
`sqlite3`, and `ipython-sql`.

Each notebook downloads its own inputs from public URLs (the SpaceX API, Wikipedia, and IBM
Cloud Object Storage), so no dataset files are checked in. Outputs are committed, so the
notebooks can be read without re-running them.

Two of the course's deliverables are not in this repo: the Plotly Dash dashboard and the
final presentation deck.
