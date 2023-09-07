import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

with open('RFclassifier', 'rb') as pickle_file:
    RFclassifier = pickle.load(pickle_file)

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

spt = pd.DataFrame().assign(Sport=df['Sport'])
spt = spt.drop_duplicates()
spt = spt.sort_values('Sport')
spt.drop(58, inplace=True, axis=0)

data = pd.read_csv('athletes(6).csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

country = data['NOC'].sort_values().unique()
sport = data['Sport'].sort_values().unique()

country_dict = {}
zero_list = []

for i in range(220):
    zero_list.append(0)
index = 0
for c in country:
    temp_list = zero_list.copy()
    temp_list[index] = 1
    index += 1
    country_dict[c] = temp_list

sport_dict = {}
zero_list = []
for i in range(65):
    zero_list.append(0)
index = 0
for s in sport:
    temp_list = zero_list.copy()
    temp_list[index] = 1
    index += 1
    sport_dict[s] = temp_list

df = preprocessor.preprocess(df, region_df)

st.sidebar.title("Olympics Analysis")
st.sidebar.image(
    'https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings'
    '-olympics-logo-text-sport.png')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis', 'Winner Prediction')
)

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(
        x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
        annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)

if user_menu == 'Country-wise Analysis':
    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    country_df = helper.yearwise_medal_tally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df, selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt, annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df, selected_country)
    st.table(top10_df)

if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

df1 = pd.read_csv('country.csv')

if user_menu == 'Winner Prediction':
    st.title('Olympic Winner Predictor')
    st.title('')

    col1, col2, col3 = st.columns(spec=3, gap="large")

    with col1:
        Country = st.selectbox(
            'Country',
            df1['Country'])

        st.header('')

        Gender = st.radio(
            'Gender: ',
            ('Male', 'Female'), horizontal=True)

    with col2:
        Sport = st.selectbox(
            'Sport',
            spt['Sport']
        )

        st.header('')

        Height = st.text_input('Height(in cms): ', value='', placeholder='Please enter the height',
                               label_visibility='visible')

    with col3:
        Age = st.text_input('Age: ', value='', placeholder='Please enter the age', label_visibility='visible')

        st.header('')

        Weight = st.text_input('Weight(in kgs): ', value='', placeholder='Please enter the weight',
                               label_visibility='visible')

    st.header('')
    predict_button = st.button('Predict')

    if predict_button:
        country_input = Country[:3]
        sport_input = Sport
        age_input = float(Age)
        gender_input = 1 if Gender[:1] == 'M' else 0
        height_input = float(Height)
        weight_input = float(Weight)
        season = 1
        year = 2024

        l = [gender_input, age_input, height_input, weight_input, year, season]

        for item in country_dict[country_input]:
            l.append(item)

        for item in sport_dict[sport_input]:
            l.append(item)

        l1 = [l]
        scaler = StandardScaler()

        l1 = scaler.fit_transform(l1)

        st.header('')
        if age_input < 18.0 or age_input >= 40 or height_input <= 140 or height_input >= 200 or weight_input <= 50 or weight_input>=85 or country_input == 'AFG':
            st.subheader('Unfortunately, the athlete is less likely to win')
        else:
            st.subheader(
                'Congratulations, the athlete is very likely to win!' if RFclassifier.predict(l1)[0] == 1 else 'Unfortunately, the athlete is less likely to win')
