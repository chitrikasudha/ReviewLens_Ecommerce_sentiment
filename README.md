# ReviewLens Multi-Store Sentiment Analytics

## Run on Windows PowerShell
Open PowerShell in the folder containing `app.py` and `requirements.txt`, then run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the local URL shown by Streamlit, usually http://localhost:8501.

## Pages
- Store & Product Explorer: select Amazon, Flipkart, Myntra, or Walmart, then select a product to see review predictions and charts.
- Analyze a Review: paste text and see sentiment and probability estimates.
- Compare Products: compare sentiment across demo stores.
- Upload Notebook / CSV Data: upload your own review CSV; supports common columns such as `reviews.text`, `reviews.title`, and `sentiment`.
- Model & Methodology: workflow and limitations.

## Important data note
Included reviews are illustrative demo examples, not live reviews from those websites. This app does not scrape marketplaces or fetch live data. Your notebook refers to `train_data.csv` and `test_data.csv`; upload those files on the CSV page to analyze the actual dataset. The bundled classifier is trained on a small demonstration dataset, so do not claim production-level accuracy from it.
