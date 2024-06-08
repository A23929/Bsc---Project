import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json
from streamlit_lottie import st_lottie
import requests
import sklearn
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from pandas.errors import ParserError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs, Slope
from bokeh.palettes import Set3
from sklearn.preprocessing import LabelBinarizer
import base64
import seaborn as sns
from scipy.stats import linregress
from sklearn.feature_selection import RFE
import statsmodels.api as sm  
import statsmodels.formula.api as smf

st.set_page_config("EDA and LR",layout="wide")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style,unsafe_allow_html=True)



class Predictor:
    # Data preparation part, it will automatically handle with your data
    def prepare_data(self, split_data, train_test):
        # Reduce data size
        data = self.data[self.features]
        data = data.sample(frac = round(split_data/100,2))

        # Impute nans with mean for numeris and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
            data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
        imp = SimpleImputer(missing_values = np.nan, strategy="mean")
        data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])

        # One hot encoding for categorical variables
        cats = data.dtypes == 'object'
        le = LabelEncoder()
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:,x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder()
        data.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(data.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names_out()))

        # Set target column
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)  

 
    # Model training and predicitons
    def predict(self, predict_btn):    

        self.alg = LinearRegression()
        self.model = self.alg.fit(self.X_train, self.y_train)
        predictions = self.alg.predict(self.X_test)
        self.predictions_train = self.alg.predict(self.X_train)
        self.predictions = predictions

        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.predictions
        result_train['Prediction_Train'] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if(True and len(self.features) == 2):
            self.error_metrics['R2_test']= r2_score(self.y_test,self.predictions)
            self.error_metrics['R2_train'] = r2_score(self.y_train,self.predictions_train)
            ss = r'$\R^2$'
            return st.markdown(ss +' Train: ' + str(round(self.error_metrics['R2_train'], 3)) + ' --- ' +ss + ' Test: ' + str(round(self.error_metrics['R2_test'], 3)))


    # Plot the predicted values and real values
    def plot_result(self):
       
        if len(controller.features) == 2 :

            output_file("slider.html")

            par = np.polyfit(self.result_train.index, self.result_train.Prediction_Train, 1, full=True)
            slope, intercept, r_value, p_value, std_err = linregress(self.result_train.index,self.result_train.Prediction_Train)
            y_predicted = [slope*i + intercept  for i in self.result_train.index]

            s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
            s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color="Black", alpha=1, legend_label = "Actual")
            s1.circle(self.result_train.index, self.result_train.Prediction_Train, size=12, color="Red", alpha=1, legend_label = "Prediction")

            s1.line(self.result_train.index, y_predicted, color='red', legend_label='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
           
            tab1 = Panel(child=s1, title="Train Data")

            if self.result.Actual is not None:
                par1 = np.polyfit(self.result.index, self.result.Prediction, 1, full=True)
                slope1, intercept1, r_value1, p_value1, std_err1 = linregress(self.result.index,self.result.Prediction)
                y_predicted1 = [slope1*i + intercept1  for i in self.result.index]

                s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
                s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1, legend_label = "Actual")
                s2.circle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1, legend_label = "Prediction")

                s2.line(self.result.index, y_predicted1, color='blue', legend_label='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
               
                tab2 = Panel(child=s2, title="Test Data")
                tabs = Tabs(tabs=[ tab1, tab2 ])
            else:
                tabs = Tabs(tabs=[ tab1])

            st.bokeh_chart(tabs)

        else:

            x_columns = list(self.features)
            x_columns.remove(str(self.chosen_target))
            w = self.data
            w = w.apply(lambda x: pd.factorize(x)[0])
            y = w[str(self.chosen_target)]

            x=w[x_columns]

            res=sm.OLS(y,x).fit()

            l = x_columns

            i=0

            while True:
               
                if i == len(x_columns):
                    break

                if res.pvalues[i] > 0.05:
                    x_columns.remove(x_columns[i])
                    x=w[x_columns]
                    res=sm.OLS(y,x).fit()          

                else:
                    i+=1

            st.text(res.summary())


    # File selector module for web app
    def file_selector(self,file):
       
        if file is not None:
            data = pd.read_csv(file)
            return data

    def print_table(self):
        if len(self.result) > 0:
            result = self.result[['Actual', 'Prediction']]
            st.dataframe(result.sort_values(by='Actual',ascending=False))
   
    def set_features(self):
        self.features = st.multiselect('Please choose the features including target variable that go into the model', self.data.columns)


# Web App Title

st.markdown("<h1 style='text-align: center; color: black;'>Automation of Linear Models</h1>", unsafe_allow_html=True)
file_bytes = fp = st.sidebar.file_uploader("Upload CSV for EDA Analysis/Linear Regression",type="csv") 

add_selectbox = st.sidebar.selectbox(
    "Overview, EDA, Simple Linear Regression and Multiple Linear Regression",
    ("Overview","EDA", "LR")
)

if file_bytes is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('''
        Welcome to our website! You might be wondering what all this is. Exploratory Data Analysis also known as EDA
        is a way of analyzing data sets to summarize their main characteristics, often using statistical graphics and 
        other data visualization methods. In statistics, linear regression is a linear approach for modelling the 
        relationship between a scalar response and one or more explanatory variables. The case of one explanatory variable 
        is called simple linear regression; for more than one, the process is called multiple linear regression.
        ''')

    with col2:
        def load_lottiefile(filepath:str):
            with open(filepath,"r") as f:
                return json.load(f)
        lottie_coding = load_lottiefile("lottiefiles/animation.json")
        st_lottie(lottie_coding,key="hello")

    with col3:
        st.write('''
            In our website you can upload any CSV file that you would like to perform an EDA on, and feel 
            free to play around with it because you never know what you may learn from your data. Navigate 
            to the`Upload CSV for EDA Analysis/Linear Regression` area to visualize your data. If you do not have any data feel free to 
            use the example dataset provided below. You will be given the option to clean your data, as well as 
            visualize you Categorical and Numerical Data.
            ''')
    st.write('''
        Data analytics is the future, and the future is NOW! Every mouse click, keyboard button press, swipe or tap is used to shape business decisions, Everything is about data these days, Data is information, and information is power.
        ''')
        
# Upload CSV data

col4, col5 = st.columns(2)

if file_bytes is not None and add_selectbox == "Overview":
    @st.cache
    def load_csv():
        csv = pd.read_csv(file_bytes)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Data Profile Report**')
    st_profile_report(pr)

if file_bytes is not None and add_selectbox == "EDA":
    data = pd.read_csv(file_bytes)
    obj = []
    int_float = []
    
    for i in data.columns:
        clas = data[i].dtypes
        if clas == "object":
            obj.append(i)
        else:
            int_float.append(i)

    #Submit button
    with st.form(key="my_form"):
        with st.sidebar:
            st.sidebar.subheader("If you would like to clean data please press Submit below")
            submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        for i in data.columns:
            clas = data[i].dtypes
            if clas=="object":
                data[i].fillna(data[i].mode()[0],inplace=True)
            else:
                data[i].fillna(data[i].mean(),inplace=True)

    lis=[]

    for i in data.columns:
        dd = sum(pd.isnull(data[i]))
        lis.append(dd)
    
    add_selectbox1 = st.sidebar.selectbox(
        "Null Values, Frequency Plots (Bar Charts), Histogram, Box Plot, Line Chart, Scatterplot",
        ("Dataframe","Null Values","Frequency Plots","Histogram","Box Plot", "Line Chart","Scatterplot")
    )   


    #Dataframe
    if add_selectbox1 == "Dataframe":
        st.header('**Input DataFrame**')
        st.write(data)

    # Null Values
    if add_selectbox1 == "Null Values":
        st.header("Bar Plot - Frequency of Null Values In Dataframe")
        st.subheader("Number of Null Values = " + str(sum(lis)))
        fig2 = px.bar(x=data.columns, y=lis, labels={'x':"Column Names",'y': "Number of Null Values"},width=1300,height=600)
        st.plotly_chart(fig2)

    # Bar Chart
    if add_selectbox1 == "Frequency Plots":
        st.sidebar.header("Select Variable for Bar Charts")
        selected_pos = st.sidebar.selectbox("Categorical Variables",obj)
        selected_pos1 = st.sidebar.selectbox("Int or Float Variables",int_float)
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("<h3 style='text-align: center; color: black;'>Bar Plot - Frequency Of Each Category</h3>", unsafe_allow_html=True)
            frequency_data = data[selected_pos].value_counts()
            fig = px.bar(frequency_data,x=frequency_data.index,y=selected_pos,labels={'x':selected_pos,'y':"count"})
            st.plotly_chart(fig)

        with col5:
            st.markdown("<h3 style='text-align: center; color: black;'>Bar Plot - Frequency Of Values</h3>", unsafe_allow_html=True)
            counts,bins = np.histogram(data[selected_pos1],bins=range(int(min(data[selected_pos1])),int(max(data[selected_pos1]))))
            bins = 0.5 * (bins[:-1]+bins[1:])
            fig1 = px.bar(x=bins,y=counts,labels={'x':selected_pos1,'y':'count'})
            st.plotly_chart(fig1)

    #Histogram
    if add_selectbox1 == "Histogram":
        selected_pos2 = st.sidebar.selectbox("X Axis",int_float)
        fig7 = px.histogram(data, x=selected_pos2,width=1300,height=700)
        st.plotly_chart(fig7)

    #Box Plot
    if add_selectbox1 == "Box Plot":
        st.sidebar.header("Select Variables for Box Plot")
        selected_pos1 = st.sidebar.selectbox("Int or Float Variables",int_float)
        selected_pos3 = st.sidebar.checkbox("With Categorical Variables")
        selected_pos2 = st.sidebar.selectbox("Categorical Variables",obj)
        st.header("BOX PLOT - RANGE OF VARIABLES")
        if selected_pos3:
            fig4 = px.box(data,x=selected_pos2,y=selected_pos1,color=selected_pos2,width=1300,height=700)
        else:
            fig4 = px.box(data,y=selected_pos1,width=1300,height=700)
        st.plotly_chart(fig4)

    #Line Chart
    if add_selectbox1 == "Line Chart":
        st.sidebar.header("Select Variables for Line Chart")
        selected_pos2 = st.sidebar.selectbox("X Axis",int_float)
        selected_pos3 = st.sidebar.selectbox("Y Axis",int_float)
        st.header("LINE CHART - " + selected_pos2 + " vs " + selected_pos3)
        fig5 = px.line(data, x=data[selected_pos2], y=data[selected_pos3],width=1300,height=700)
        st.plotly_chart(fig5)

    

    #Scatterplot
    if add_selectbox1 == "Scatterplot":
        st.sidebar.header("Select Variables for Scatterplot")
        selected_pos2 = st.sidebar.selectbox("X Axis",int_float)
        selected_pos3 = st.sidebar.selectbox("Y Axis",int_float)
        selected_pos4 = st.sidebar.checkbox("With Categorical Variables")
        selected_pos5 = st.sidebar.selectbox("Categorical Variables",obj)
        st.header("Scatterplot - " + selected_pos2 + " w " + selected_pos3)
        if selected_pos4:
            fig6 = px.scatter(data,x=data[selected_pos2],y=data[selected_pos3],color=selected_pos5,width=1300,height=700)
        else:
            fig6 = px.scatter(data, x=data[selected_pos2], y=data[selected_pos3],width=1300,height=700)
        st.plotly_chart(fig6)

#LR

if file_bytes is not None and add_selectbox == "LR":

    controller = Predictor()
    try:
        controller.data = controller.file_selector(file_bytes)

        if controller.data is not None:
            split_data = st.sidebar.slider('Data Size Used %', 1, 100, 100 )
            train_test = st.sidebar.slider('Train-test split %', 1, 99, 85 )


        controller.set_features()

        if len(controller.features) > 1:
            controller.prepare_data(split_data, train_test)
            predict_btn = st.sidebar.button('Predict')  

    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)  

    except (ValueError) as f:
        pass

    if controller.data is not None and len(controller.features) > 1:
        try:

            if predict_btn:
                st.sidebar.text("Progress:")
                my_bar = st.sidebar.progress(0)
                predictions, predictions_train, result, result_train = controller.predict(predict_btn)
                for percent_complete in range(100):
                    my_bar.progress(percent_complete + 1)
               
                if len(controller.features) >= 2:
                    controller.get_metrics()        

                controller.plot_result()
                if len(controller.features) == 2:
                    controller.print_table()

                data = controller.result.to_csv(index=False)
                b64 = base64.b64encode(data.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.sidebar.markdown(href, unsafe_allow_html=True)



        except (NameError):
            st.markdown('<span style="color:blue">At least one numerical feature data type required</span>', unsafe_allow_html=True)

    if controller.data is not None:
        if st.sidebar.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(controller.data)   