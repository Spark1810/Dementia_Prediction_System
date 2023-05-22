import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image

sns.set()

st.title("DEMENTIA PREDICTION ")

activities = ["Introduction", "Statistics", "Prediction", "Dementia Report", "About Us"]
choice = st.sidebar.selectbox("Select Activities", activities) 

if choice == 'Introduction':
    st.markdown(
        "Dementia is a term used to describe a group of symptoms affecting memory, thinking and social abilities severely enough to interfere with your daily life. It isn't a specific disease, but several diseases can cause dementia. Though dementia generally involves memory loss, memory loss has different causes")
    st.title("A look into the scientific side of demenetia ")
    st.write("Parameters taken")
    st.write("A major parameters for dementia prediction is MMSE,SES,eTIV,nWBV,ASF")
    st.write("MMSE - Mini Mental State Examination")
    st.write("SES - Social Economic State")
    st.write("eTIV - Estimated Total Intracranial Volume")
    st.write("nWBV - Normalised  Whole Brain Volume")
    st.write("ASF - Atlas Scaling Factor")
    st.write("Each one of those parameters have a particular effect when predicting dementia.")
    
# ==========================================================================================================================

elif choice == 'Statistics':
    import matplotlib.pyplot as plt

    st.title("Wanna Clarify about your Dementia status ?")
    df = pd.read_csv(r"oasis_longitudinal.csv")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # ================================================
    df = df.loc[df['Visit'] == 1]
    # use first visit data only because of the analysis
    df = df.reset_index(drop=True)
    # reset index after filtering first visit data
    df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
    # Male/Female column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
    # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)


    def bar(feature):
        Demented = df[df['Group'] == 1][feature].value_counts()
        Nondemented = df[df['Group'] == 0][feature].value_counts()
        _bar = pd.DataFrame([Demented, Nondemented])
        _bar.index = ['Demented', 'Nondemented']
        _bar.plot(kind='bar', stacked=True, figsize=(8, 5))


    # Gender  and  Group ( Female=0, Male=1)
    bar('M/F')
    plt.xlabel('Group')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.title('Gender v/s Demented rate')
    # =================================================================
    # Create a bar chart using the value_counts() method on the 'M/F' column of the DataFrame
    dementia_by_gender = df[df['Group'] == 1]['M/F'].value_counts()
    dementia_by_gender.plot(kind='bar')
    # Set the title and axis labels
    st.subheader('Dementia Distribution by Gender :')
    plt.title('Dementia Distribution by Gender')
    plt.xlabel('Gender (Female=0, Male=1)')
    plt.ylabel('Number of Patients')
    # Display the chart in Streamlit
    st.pyplot()
    # =====================================================================
    # MMSE : Mini Mental State Examination
    st.subheader('Dementia Distribution by MMSE :')
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'MMSE', shade=True)
    facetgrid.set(xlim=(0, df['MMSE'].max()))
    facetgrid.add_legend()
    plt.xlim(16.00)
    st.pyplot()
    # Graph on each variable
    st.subheader('Dementia Distribution by ASF :')
    # bar_chart('ASF') = Atlas Scaling Factor
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'ASF', shade=True)
    facetgrid.set(xlim=(0, df['ASF'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 1.8)
    st.pyplot()
    st.subheader('Dementia Distribution by ETIV :')
    # eTIV = Estimated Total Intracranial Volume
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'eTIV', shade=True)
    facetgrid.set(xlim=(0, df['eTIV'].max()))
    facetgrid.add_legend()
    plt.xlim(900, 2200)
    st.pyplot()
    st.subheader('Dementia Distribution by nWBV :')
    # 'nWBV' = Normalized Whole Brain Volume
    # Nondemented = 0, Demented =1
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'nWBV', shade=True)
    facetgrid.set(xlim=(0, df['nWBV'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 0.9)
    st.pyplot()
    st.subheader('Dementia Distribution by AGE :')
    # AGE.
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'Age', shade=True)
    facetgrid.set(xlim=(0, df['Age'].max()))
    facetgrid.add_legend()
    plt.xlim(50, 110)
    st.pyplot()
    # st.title('Dementia Distribution by YEARS OF EDUCATION :')
    # 'EDUC' = Years of Education
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'EDUC', shade=True)
    facetgrid.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
    facetgrid.add_legend()
    plt.ylim(0, 0.16)
    plt.xlim(2, 25)
    df.isnull().sum()
    df = df.dropna(axis=0, how="any")
    pd.isnull(df).sum()
    df['Group'].value_counts()
    st.subheader('Dementia Distribution by YEARS OF EDUCATION AND SES :')
    # Draw scatter plot between EDUC and SES
    x = df['EDUC']
    y = df['SES']
    ses_not_null = y[~y.isnull()].index
    x = x[ses_not_null]
    y = y[ses_not_null]
    # Trend line
    poly = np.polyfit(x, y, 1)
    pp = np.poly1d(poly)
    plt.plot(x, y, 'go', x, pp(x), "b--")
    plt.xlabel('Education Level(EDUC)')
    plt.ylabel('Social Economic Status(SES)')
    st.pyplot()
    plt.show()
    
    # ======================================================================

elif choice == 'Prediction':

    st.title("Check your Dementia status...")
    df = pd.read_csv(r"oasis_longitudinal.csv")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # ================================================
    df = df.loc[df['Visit'] == 1]
    # use first visit data only because of the analysis
    df = df.reset_index(drop=True)
    # reset index after filtering first visit data
    df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
    # Male/Female column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
    # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)


    def bar(feature):
        Demented = df[df['Group'] == 1][feature].value_counts()
        Nondemented = df[df['Group'] == 0][feature].value_counts()
        _bar = pd.DataFrame([Demented, Nondemented])
        _bar.index = ['Demented', 'Nondemented']
        _bar.plot(kind='bar', stacked=True, figsize=(8, 5))


    # Gender  and  Group ( Female=0, Male=1)
    bar('M/F')
    plt.xlabel('Group')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.title('Gender v/s Demented rate')
    # =================================================================
    # Create a bar chart using the value_counts() method on the 'M/F' column of the DataFrame
    dementia_by_gender = df[df['Group'] == 1]['M/F'].value_counts()
    dementia_by_gender.plot(kind='bar')
    # Set the title and axis labels
    plt.title('Dementia Distribution by Gender')
    plt.xlabel('Gender (Female=0, Male=1)')
    plt.ylabel('Number of Patients')
    # Display the chart in Streamlit
    # st.pyplot()
    # =====================================================================
    # MMSE : Mini Mental State Examination
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'MMSE', shade=True)
    facetgrid.set(xlim=(0, df['MMSE'].max()))
    facetgrid.add_legend()
    plt.xlim(16.00)
    # st.pyplot()
    # Graph on each variable
    # bar_chart('ASF') = Atlas Scaling Factor
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'ASF', shade=True)
    facetgrid.set(xlim=(0, df['ASF'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 1.8)
    # st.pyplot()
    # eTIV = Estimated Total Intracranial Volume
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'eTIV', shade=True)
    facetgrid.set(xlim=(0, df['eTIV'].max()))
    facetgrid.add_legend()
    plt.xlim(900, 2200)
    # st.pyplot()
    # 'nWBV' = Normalized Whole Brain Volume
    # Nondemented = 0, Demented =1
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'nWBV', shade=True)
    facetgrid.set(xlim=(0, df['nWBV'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 0.9)
    # st.pyplot()
    # AGE.
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'Age', shade=True)
    facetgrid.set(xlim=(0, df['Age'].max()))
    facetgrid.add_legend()
    plt.xlim(50, 110)
    # st.pyplot()
    # 'EDUC' = Years of Education
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'EDUC', shade=True)
    facetgrid.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
    facetgrid.add_legend()
    plt.ylim(0, 0.16)
    plt.xlim(2, 25)
    df.isnull().sum()
    df = df.dropna(axis=0, how="any")
    pd.isnull(df).sum()
    df['Group'].value_counts()
    # Draw scatter plot between EDUC and SES
    x = df['EDUC']
    y = df['SES']
    ses_not_null = y[~y.isnull()].index
    x = x[ses_not_null]
    y = y[ses_not_null]
    # Trend line
    poly = np.polyfit(x, y, 1)
    pp = np.poly1d(poly)
    plt.plot(x, y, 'go', x, pp(x), "b--")
    plt.xlabel('Education Level(EDUC)')
    plt.ylabel('Social Economic Status(SES)')
    # st.pyplot()
    # plt.show()

    # ============================================================================================================================================    PREDICTION

    gender = st.sidebar.selectbox(
        "Gender",
        ("Female", "Male")
    )
    gender = 1 if gender == "Male" else 2
    age = st.sidebar.selectbox(
        "Age",
        ('18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64',
         '65 to 69', '70 to 74', '75 to 79', '80 or older')
    )
    if age == "18 to 24":
        age = 1
    elif age == "25 to 29":
        age = 2
    elif age == "30 to 34":
        age = 3
    elif age == "35 to 39":
        age = 4
    elif age == "40 to 44":
        age = 5
    elif age == "45 to 49":
        age = 6
    elif age == "50 to 54":
        age = 7
    elif age == "55 to 59":
        age = 8
    elif age == "60 to 64":
        age = 9
    elif age == "65 to 69":
        age = 10
    elif age == "70 to 74":
        age = 11
    elif age == "80 or older":
        age = 12
    else:
        age = 13
    EDUC = st.sidebar.slider("Years of Education", max_value=30)
    MMSE = st.sidebar.slider("MMSE Value", max_value=40)
    SES = st.sidebar.slider("SES Value", max_value=10)
    eTIV = st.sidebar.slider("eTIV Value", max_value=2040)
    nWBV = st.sidebar.number_input("nWBV Value")
    ASF = st.sidebar.number_input("ASF Value")
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.impute import SimpleImputer

    a = pd.read_csv(r"oasis_longitudinal.csv")
    a.isna().sum()
    # create the object of the imputer for null ratings
    im = SimpleImputer(strategy='mean')
    # im= SimpleImputer(strategy= 'most_frequent')
    # fit the ratings imputer with the data and transform
    im.fit(a[['MMSE']])
    a[['MMSE']] = im.transform(a[['MMSE']])
    # create the object of the imputer for null ratings
    im = SimpleImputer(strategy='mean')
    # im= SimpleImputer(strategy= 'most_frequent')
    # fit the ratings imputer with the data and transform
    im.fit(a[['SES']])
    a[['SES']] = im.transform(a[['SES']])
    x = a.iloc[:, 7:-1].values
    y = a.iloc[:, -1].values
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    asf = regressor.predict([[87, 13, 2.0, 24, 0.5, 2000, 0.736]])  # for asf
    print(asf)
    # if asf>:
    #    print("Non Dementiated")
    # else:
    #    print("Dementiated")
    x1 = a.iloc[:, 11:14].values
    y1 = a.iloc[:, 10:11].values
    from sklearn.model_selection import train_test_split

    X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=1 / 3, random_state=0)
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train1, y_train1)
    y_pred1 = regressor.predict(X_test1)
    mmse = regressor.predict([[nWBV, eTIV, SES]])  # for mmse
    print(mmse)
    print(nWBV, eTIV, SES)
    if mmse > 26.77:
        st.success('Your Result : You are Non Dementiated', icon="✅")
        st.balloons()
        re = "Non Dementiated"
    else:
        re = "Dementiated"
        st.warning('Your Result : You are Dementiated', icon="⚠️")
        from plyer import notification
        import time

        # Define the notification message and icon
        title = 'Warning'
        message = 'Your Result : You are Dementiated'
        # Show the notification with the animated icon
        notification.notify(
            title=title,
            message=message,
            timeout=5,
            toast=True
        )
        # Keep the notification displayed for 5 seconds
        for i in range(5):
            time.sleep(1)
    import mysql.connector

    con = mysql.connector.connect(host='localhost', user='root', passwd='', database='')
    mycursor = con.cursor()
    try:
        mycursor.execute("CREATE DATABASE demen")
        mycursor.execute("USE demen")
        print("Database created")
    except:
        mycursor.execute("USE demen")
        print("Database Enhanced")
    try:
        command = "create table udb2(userid varchar(100),gender varchar(10),age int(5),educa int (30),mmse int(30),ses int(10),etiv int(10),nwbv varchar(20),asf varchar(30),result varchar(30));"
        mycursor.execute(command)
        print("Table has been created")
    except:
        print("TABLE ENHANCED")
    if st.button('Save To Database'):
        gender = "M" if gender == 1 else "F"
        mycursor.execute(
            "INSERT INTO udb2 VALUES('{}','{}',{},{},{},{},{},'{}','{}','{}')".format("OAS2_0001", gender, int(age),
                                                                                      int(EDUC), int(MMSE), int(SES),
                                                                                      int(eTIV), str(nWBV), str(ASF),
                                                                                      re))
        con.commit()
        st.write('Saved to Database ')
    if st.button('View Database'):
        st.write('Recent Data : ')
        import mysql.connector

        con = mysql.connector.connect(host='localhost', user='root', passwd='', database='')
        mycursor = con.cursor()
        import pandas as pd
        import streamlit as st

        mycursor.execute("USE demen")
        mysql = "select*from udb2"
        mycursor.execute(mysql)
        data = mycursor.fetchall()
        df3 = pd.DataFrame({
            'PATIENT ID': [],
            'GENDER': [],
            'AGE GROUP': [],
            'YEARS OF EDUCATION': [],
            'MMSE': [],
            'SES': [],
            'ETIV': [],
            'NWBV': [],
            'ASF': [],
            'RESULT': []
        })

        # EMPTY TABLE VIEW
        # st.table(df3)
        for data2 in data:
            new_row = {"PATIENT ID": data2[0], "GENDER": data2[1], "AGE GROUP": data2[2],
                       "YEARS OF EDUCATION": data2[3], "MMSE": data2[4], "SES": data2[5], "ETIV": data2[6],
                       "NWBV": data2[7], "ASF": data2[8], "RESULT": data2[9]}
            df3 = df3.append(new_row, ignore_index=True)
        # Display the updated dataframe as a table in Streamlit
        st.table(df3)

# ===============================================================================

elif choice == 'Dementia Report':
    import matplotlib.pyplot as plt

    st.title("DEMENTIA REPORT ")

    import mysql.connector

    con = mysql.connector.connect(host='localhost', user='root', passwd='', database='')
    mycursor = con.cursor()
    import pandas as pd
    import streamlit as st

    mycursor.execute("USE demen")
    mysql = "select*from udb2"
    mycursor.execute(mysql)
    data = mycursor.fetchall()
    df3 = pd.DataFrame({
        'PATIENT ID': [],
        'GENDER': [],
        'AGE GROUP': [],
        'YEARS OF EDUCATION': [],
        'MMSE': [],
        'SES': [],
        'ETIV': [],
        'NWBV': [],
        'ASF': [],
        'RESULT': []
    })
    # EMPTY TABLE VIEW
    # st.table(df3)
    for data2 in data:
        new_row = {"PATIENT ID": data2[0], "GENDER": data2[1], "AGE GROUP": data2[2], "YEARS OF EDUCATION": data2[3],
                   "MMSE": data2[4], "SES": data2[5], "ETIV": data2[6], "NWBV": data2[7], "ASF": data2[8],
                   "RESULT": data2[9]}
        df3 = df3.append(new_row, ignore_index=True)
    # Display the updated dataframe as a table in Streamlit
    # st.table(df3)

    import streamlit as st
    import pandas as pd
    import bamboolib as bam
    import plotly.express as px

    # Create a bamboolib dataframe
    df3
    bam.enable()

    # Streamlit components
    st.header('Data Exploration and Manipulation')
    st.subheader(' ')

    # Add data exploration and manipulation steps using bamboolib
    col1, col2 = st.columns(2)

    # Add checkboxes to the first column
    with col1:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.checkbox("Correlation Plot"):
            plt.matshow(df3.corr())
            st.pyplot()

    # Add the filtered DataFrame to the second column
    with col2:
        if st.checkbox("Summary"):
            st.write(df3.describe())
    st.subheader(' ')
    st.subheader('Filtering Columns ')

    selected_column = st.selectbox('Select Column', df3.columns)

    # ======================================================================

    unique_values = df3[selected_column].unique()

    # Create a multiselect dropdown with the unique values
    selected_values = st.multiselect('Select Values to Filter ', unique_values)

    # Filter the DataFrame based on the selected values
    filtered_df = df3[df3[selected_column].isin(selected_values)]

    # Display the filtered DataFrame
    st.write(filtered_df)
    st.subheader(' ')
    st.subheader('Aggregation Function ')

    selected_function = st.selectbox('Select Function', ['Mean', 'Sum', 'Max', 'Min'])

    # Display the results
    st.subheader(f'{selected_function} of {selected_column}')
    st.write(df3.groupby(selected_column).agg(selected_function.lower()))

    # ======================================================================
    st.subheader(' ')
    st.subheader('Visualizations ')
    # Add data exploration and manipulation steps using bamboolib
    col3, col4 = st.columns(2)

    # Add checkboxes to the first column
    with col3:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        column = st.selectbox('Select column to plot Y axis', df3.columns)
        column2 = st.selectbox('Select column to plot X axis', df3.columns)
        chart_type = st.radio('Select Chart Type', ('Line Chart', 'Bar Chart', 'Scatter Plot', 'Pie Chart'))
        if chart_type == 'Line Chart':
            chart = px.line(df3, x=column2, y=column)
        elif chart_type == 'Bar Chart':
            chart = px.bar(df3, x=column2, y=column)
        elif chart_type == 'Scatter Plot':
            chart = px.scatter(df3, x=column2, y=column)
        elif chart_type == 'Pie Chart':
            chart = px.pie(df3, values=column2, names=column)

    # Add the filtered DataFrame to the second column
    with col4:
        st.plotly_chart(chart)

        
# ============================================================================================
elif choice == "About Us":
    st.write("CREATED BY SURYA RAJIV KUMAR AND SUDHARSHAN VIJAY SK")

