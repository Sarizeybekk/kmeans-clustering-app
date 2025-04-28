# KMeans Customer Segmentation App

An interactive web application built with **Streamlit** for customer segmentation using **K-Means Clustering**.  
The app evaluates cluster quality using the **Davies-Bouldin Index** and provides visual insights into optimal customer groupings.

---

## Features

- Upload your own customer CSV dataset
- Choose which features (columns) to use
- Select the number of clusters (K)
- Visualize customer clusters in a scatter plot
- View Davies-Bouldin Score on a gauge chart
- Simple and intuitive Streamlit interface

---

##  Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Plotly

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/Sarizeybekk/kmeans-clustering-app.git
cd kmeans-clustering-app
```

```bash
python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
# venv\Scripts\activate   # For Windows (use this instead)
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```
![image](https://github.com/user-attachments/assets/7aa79c68-d2d5-4d81-b6f5-91164cd40b44)
