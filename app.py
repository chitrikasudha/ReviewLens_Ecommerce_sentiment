import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title='ReviewLens Multi-Store Sentiment', page_icon='🛍️', layout='wide')
COLORS={'Negative':'#ff5b65','Neutral':'#f5b942','Positive':'#19c6b4'}
LABELS=['Negative','Neutral','Positive']
DATA=Path(__file__).parent/'data'/'demo_reviews.csv'
def clean(text):
    text=str(text).lower()
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    return re.sub(r'\s+',' ',re.sub(r"[^a-z0-9\s']",' ',text)).strip()
@st.cache_resource
def train_model():
    d=pd.read_csv(DATA)
    pipe=Pipeline([('tfidf',TfidfVectorizer(preprocessor=clean,ngram_range=(1,2),max_features=10000,sublinear_tf=True)),('classifier',LogisticRegression(max_iter=1200,class_weight='balanced',random_state=42))])
    pipe.fit(d.Review,d.Sentiment)
    return pipe
model=train_model()
def classify(text):
    probs=model.predict_proba([str(text)])[0]; i=int(probs.argmax())
    return model.classes_[i],float(probs[i]),{model.classes_[j]:float(probs[j]) for j in range(len(probs))}
def topics(text):
    t=str(text).lower(); groups={'Quality & durability':['quality','broken','broke','durable','damage','material','defect','build'],'Price & value':['price','cost','expensive','cheap','value','money','worth'],'Performance & features':['performance','battery','speed','fast','slow','feature','sound','screen','camera','wifi'],'Delivery & packaging':['delivery','arrived','shipping','package','packaging','late','box'],'Fit & usability':['fit','size','comfortable','easy to use','difficult','setup','design']}
    return [k for k,words in groups.items() if any(w in t for w in words)]
def normalize_columns(df):
    df=df.copy(); lower={str(c).strip().lower():c for c in df.columns}
    def find(cands):
        return next((lower[c] for c in cands if c in lower),None)
    rc=find(['review','reviews.text','reviews_text','review text','review_text','text','content','reviewdescription','review_description'])
    if rc is None:return None,'Could not find review text. Add a column named Review or reviews.text.'
    out=pd.DataFrame(); out['Review']=df[rc].fillna('').astype(str)
    tc=find(['reviews.title','reviews_title','review title','review_title','summary','title'])
    if tc and tc!=rc:out['Review']=df[tc].fillna('').astype(str)+' — '+out.Review
    pc=find(['product','product_name','name','brand','primary_categories','categories','category'])
    sc=find(['source','website','platform','store','ecommerce_website'])
    rat=find(['rating','rate','reviews.rating','overall','stars'])
    sent=find(['sentiment','label','polarity'])
    out['Product']=df[pc].fillna('Unspecified product').astype(str) if pc else 'Uploaded product'
    out['Website']=df[sc].fillna('Uploaded dataset').astype(str) if sc else 'Uploaded dataset'
    if rat:out['Rating']=pd.to_numeric(df[rat],errors='coerce')
    if sent:out['Provided sentiment']=df[sent].astype(str).str.title()
    return out,None
def add_predictions(data):
    out=data.copy(); p=out.Review.apply(classify)
    out['Predicted sentiment']=[x[0] for x in p]; out['Model probability (%)']=[round(x[1]*100,1) for x in p]
    out['Topics mentioned']=[', '.join(topics(x)) or 'General experience' for x in out.Review]
    return out
st.markdown('''<style>.stApp{background:#0b1020;color:#eef2ff}[data-testid="stSidebar"]{background:#111a2b;border-right:1px solid #27364c}.block-container{max-width:1500px;padding-top:1.2rem}h1,h2,h3{color:#f5f7ff!important}.kpi{background:linear-gradient(135deg,#19263a,#121c2e);border:1px solid #2b3a50;border-radius:14px;padding:16px;min-height:100px}.lab{font-size:11px;color:#9babc2;font-weight:700;letter-spacing:1px}.val{font-size:27px;color:#f6f8ff;font-weight:800;margin-top:7px}</style>''',unsafe_allow_html=True)
demo=pd.read_csv(DATA)
with st.sidebar:
    st.markdown('## 🛍️ ReviewLens'); st.caption('MULTI-STORE REVIEW INTELLIGENCE')
    page=st.radio('WORKSPACE',['Store & Product Explorer','Analyze a Review','Compare Products','Upload Notebook / CSV Data','Model & Methodology'])
    st.divider(); st.caption('Demo platforms'); st.write('Amazon · Flipkart · Myntra · Walmart')
    st.caption('Demo reviews are illustrative, not live marketplace data.')
def heading(t,s):st.markdown(f'<h1 style="margin-bottom:0">{t}</h1><p>{s}</p>',unsafe_allow_html=True)
def kpi(col,label,value):col.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{value}</div></div>',unsafe_allow_html=True)
if page=='Store & Product Explorer':
    heading('Multi-Store Product Sentiment Dashboard','Choose a shopping website and product to explore reviews and sentiment analytics.')
    store=st.selectbox('E-commerce website',['Amazon','Flipkart','Myntra','Walmart']); sd=demo[demo.Website==store]; products=sorted(sd.Product.unique())
    product=st.selectbox('Select a product',products); raw=sd[sd.Product==product].copy(); out=add_predictions(raw); n=len(out); counts=out['Predicted sentiment'].value_counts(); pos=int(counts.get('Positive',0)); neu=int(counts.get('Neutral',0)); neg=int(counts.get('Negative',0))
    st.markdown(f'### 🛒 {product} · {store}'); st.caption('Sample review records bundled with this prototype; not fetched from the selected marketplace.')
    a,b,c,d=st.columns(4); kpi(a,'REVIEWS ANALYZED',n); kpi(b,'POSITIVE SENTIMENT',f'{round(pos/n*100) if n else 0}%'); kpi(c,'NEUTRAL REVIEWS',neu); kpi(d,'NEGATIVE REVIEWS',neg)
    left,right=st.columns(2); dist=out['Predicted sentiment'].value_counts().reindex(LABELS,fill_value=0).rename_axis('Sentiment').reset_index(name='Reviews')
    with left:
        st.markdown('### Sentiment distribution'); fig=px.bar(dist,x='Sentiment',y='Reviews',color='Sentiment',color_discrete_map=COLORS,text='Reviews'); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False); st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown('### Sentiment share'); fig=px.pie(out,names='Predicted sentiment',hole=.62,color='Predicted sentiment',color_discrete_map=COLORS); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### Product topics mentioned'); names=['Quality & durability','Price & value','Performance & features','Delivery & packaging','Fit & usability']; td=pd.DataFrame({'Topic':names,'Mentions':[sum(topic in topics(x) for x in out.Review) for topic in names]}).sort_values('Mentions')
    fig=px.bar(td,x='Mentions',y='Topic',orientation='h',color='Mentions',color_continuous_scale=['#373b83','#19bfd0']); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### Individual customer reviews'); filt=st.multiselect('Filter by sentiment',LABELS,default=LABELS); visible=out[out['Predicted sentiment'].isin(filt)]; cols=[x for x in ['Website','Product','Rating','Review','Predicted sentiment','Model probability (%)','Topics mentioned'] if x in visible.columns]; st.dataframe(visible[cols],use_container_width=True,hide_index=True); st.download_button('Download product analysis',visible.to_csv(index=False).encode(),f'{product.lower().replace(" ","_")}_sentiment.csv','text/csv')
elif page=='Analyze a Review':
    heading('Analyze an Individual Review','Paste a review to see sentiment, probability estimates, and the model workflow.')
    examples={'Positive':'The quality is excellent and the battery lasts all day. Great value for money.','Neutral':'The package arrived yesterday and includes a charging cable.','Negative':'The product broke after two days, the battery is terrible, and delivery was late.'}; example=st.selectbox('Try an example',['Write my own',*examples]); text=st.text_area('Review text',value=examples.get(example,''),height=130)
    if st.button('Analyze sentiment',type='primary',use_container_width=True):
        if not text.strip():st.warning('Enter a review first.')
        else:
            label,conf,probs=classify(text); a,b,c=st.columns(3); a.metric('Predicted sentiment',label); b.metric('Model probability',f'{conf*100:.1f}%'); c.metric('Word count',len(text.split())); st.progress(conf)
            st.markdown('### How the review is analyzed'); st.write('1. **Preprocessing:** normalize case, links, punctuation, and spacing.'); st.write('2. **TF-IDF:** convert words and word pairs into numeric features.'); st.write('3. **Logistic Regression:** estimate probabilities for each sentiment class.'); st.write(f'4. **Prediction:** the highest estimated class is **{label}**.')
            pdp=pd.DataFrame({'Sentiment':list(probs.keys()),'Probability (%)':[round(v*100,1) for v in probs.values()]}); fig=px.bar(pdp,x='Sentiment',y='Probability (%)',color='Sentiment',color_discrete_map=COLORS,text_auto='.1f'); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False); st.plotly_chart(fig,use_container_width=True); st.markdown('**Product topics mentioned:** '+(', '.join(topics(text)) or 'No predefined topic keyword detected')); st.caption('Model probability is an estimate, not a guarantee. The bundled training data is small and for demonstration only.')
elif page=='Compare Products':
    heading('Compare Products Across Stores','Compare sentiment patterns across the demo catalogue.'); stores=st.multiselect('Stores',['Amazon','Flipkart','Myntra','Walmart'],default=['Amazon','Flipkart','Myntra','Walmart']); out=add_predictions(demo[demo.Website.isin(stores)])
    if len(out):
        counts=out.groupby(['Product','Predicted sentiment']).size().reset_index(name='Reviews'); fig=px.bar(counts,x='Product',y='Reviews',color='Predicted sentiment',barmode='group',color_discrete_map=COLORS); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_tickangle=-25); st.plotly_chart(fig,use_container_width=True); st.dataframe(out.groupby(['Website','Product','Predicted sentiment']).size().reset_index(name='Review count'),use_container_width=True,hide_index=True)
    else:st.info('Select at least one store.')
elif page=='Upload Notebook / CSV Data':
    heading('Upload Your Notebook Dataset','Analyze your Amazon notebook dataset or another permitted e-commerce review CSV.')
    st.info('Accepted review columns include **reviews.text**, **reviews_text**, **Review**, **text**, and **review_text**. Optional columns like sentiment, reviews.title, brand, categories, rating, or website are detected.')
    st.download_button('Download CSV template',pd.DataFrame({'Website':['Amazon','Flipkart'],'Product':['Kindle','Smartphone'],'Rating':[5,1],'Review':['Excellent product and easy to use.','Poor quality and stopped working.']}).to_csv(index=False).encode(),'review_template.csv','text/csv')
    file=st.file_uploader('Upload CSV file',type=['csv'])
    if file:
        try:
            norm,error=normalize_columns(pd.read_csv(file))
            if error:st.error(error)
            else:
                st.success(f'Loaded {len(norm):,} rows.'); store=st.selectbox('Filter website',sorted(norm.Website.unique())); subset=norm[norm.Website==store]; product=st.selectbox('Filter product',sorted(subset.Product.unique())); subset=subset[subset.Product==product]; out=add_predictions(subset)
                if 'Provided sentiment' in out:st.caption('Uploaded sentiment labels are retained alongside model predictions for comparison.')
                a,b,c,d=st.columns(4); a.metric('Reviews',len(out))
                for col,label in [(b,'Positive'),(c,'Neutral'),(d,'Negative')]:col.metric(label,int((out['Predicted sentiment']==label).sum()))
                dist=out['Predicted sentiment'].value_counts().reindex(LABELS,fill_value=0).rename_axis('Sentiment').reset_index(name='Reviews'); fig=px.bar(dist,x='Sentiment',y='Reviews',color='Sentiment',color_discrete_map=COLORS); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False); st.plotly_chart(fig,use_container_width=True); st.dataframe(out,use_container_width=True,hide_index=True); st.download_button('Download analyzed CSV',out.to_csv(index=False).encode(),'analyzed_reviews.csv','text/csv')
        except Exception as e:st.error(f'Could not read this file: {e}')
else:
    heading('Model & Methodology','The workflow from product selection to sentiment analytics.'); st.markdown('**Choose store → Select product → Load review text → Preprocess → TF-IDF → Logistic Regression → Sentiment dashboard**')
    st.markdown('### Platforms in demo catalogue'); st.dataframe(demo.groupby('Website').agg(Products=('Product','nunique'),Demo_reviews=('Review','count')).reset_index(),use_container_width=True,hide_index=True)
    st.markdown('### Connection to your notebook'); st.write('Your notebook describes Amazon product reviews and columns such as `reviews.text`, `reviews.title`, and `sentiment`, and it removes duplicate rows. The upload page recognizes these common column names.')
    st.warning('The included reviews are illustrative examples, not live data from Amazon, Flipkart, Myntra, or Walmart. This app does not scrape these websites. Your notebook references train_data.csv and test_data.csv, which are separate from the notebook; upload your actual CSV files to analyze them. The demo model uses a small dataset and is not suitable for production accuracy claims.')
