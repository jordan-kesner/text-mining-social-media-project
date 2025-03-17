import streamlit as st
import pandas as pd



# Set up page config
st.set_page_config(page_title="Social Media: Free Speech or Hate Speech?", layout="wide")


# Function to load CSV files
@st.cache_data
def load_data(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

# Load datasets from CSVs
newsapi_df_raw = load_data("newsapi_raw_df.csv")
reddit_df_raw = load_data("reddit_raw_df.csv")
cnn_df_raw = load_data("cnn_raw_df.csv")
foxnews_df_raw = load_data("foxnews_raw_df.csv")
newsapi_cleaned_df = load_data("newsapi_cleaned_df.csv")
reddit_cleaned_df = load_data("reddit_cleaned_df.csv")
cnn_cleaned_df = load_data("cnn_cleaned_df.csv")
foxnews_cleaned_df = load_data("foxnews_cleaned_df.csv")
newsapi_countvectorizer_df = load_data("NewsAPI_count_vectorized.csv")
newsapi_tfidfvectorizer_df = load_data("NewsAPI_tfidf_vectorized.csv")
newsapi_stemmed_df = load_data("NewsAPI_stemmed.csv")
newsapi_lemmatized_df = load_data("NewsAPI_lemmatized.csv")
merged_df = load_data("cleaned_merged_df.csv")


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data","About Me","Clustering"])

# Introduction Page
if page == "Introduction":
    st.title("Analyzing Online Discourse: Free Speech vs. Hate Speech Regulation")
    st.write("""
Social media platforms have transformed how people communicate, making them central to public discourse. While these platforms provide unprecedented opportunities for expression, they also raise concerns about the spread of harmful content. Supporters of absolute free speech argue that online platforms should allow all viewpoints, emphasizing that limiting speech on private platforms undermines democratic principles. However, proponents of content moderation highlight the dangers of misinformation, hate speech, and harassment, arguing that unchecked speech can lead to real-world harm. The tension between free expression and platform regulation has led to controversies over censorship, misinformation policies, and political bias in moderation decisions. High-profile cases, such as the banning of former U.S. President Donald Trump from Twitter and Facebook, have intensified debates about whether companies should control speech on their platforms. Some view content moderation as a necessary safeguard against extremism and misinformation, while others see it as an overreach of corporate and government power. The role of artificial intelligence in moderating content further complicates the issue, as AI models can misinterpret context, flagging satire or political speech as harmful. Global differences in free speech laws also impact how companies enforce moderation policies, creating inconsistencies across platforms. As digital spaces become the primary arenas for political and cultural debates, the question of how to balance free speech with responsible moderation remains highly contested.

Defining the boundaries of free speech online presents significant legal, ethical, and technological challenges. In the United States, the First Amendment protects most forms of speech from government censorship, but private companies are not legally required to uphold these protections. In contrast, countries such as Germany and France enforce strict hate speech laws, requiring platforms to remove harmful content within hours or face penalties. As governments explore potential regulations, some advocate for reforming Section 230, a U.S. law that shields social media companies from liability for user-generated content. Meanwhile, alternative platforms, such as Truth Social and Parler, have emerged as spaces promoting unrestricted speech, raising questions about how decentralized media environments shape public discourse. The increasing reliance on AI-driven moderation systems also introduces new concerns about bias, accuracy, and transparency in decision-making. While AI can process vast amounts of content at scale, human intervention is often required to ensure fairness in enforcement. The extent to which content moderation influences elections, social movements, and political polarization remains an open question. As digital platforms continue to evolve, determining whether free speech and content moderation can coexist—or whether one will inevitably restrict the other—remains a critical issue for governments, corporations, and society as a whole.

Key Questions to Explore:
Should social media companies be legally required to allow all forms of speech?
Does content moderation violate free speech rights, or is it necessary to prevent harm?
Who decides what qualifies as hate speech versus controversial but legal speech?
How effective are AI-based content moderation systems, and what are their limitations?
Do content moderation policies disproportionately impact specific political groups?
Should users have more control over content visibility, or should platforms dictate moderation?
How do different countries’ legal frameworks shape online speech policies?
What are the consequences of banning high-profile individuals from social media?
How do alternative social media platforms affect the broader free speech debate?
What role should governments play in regulating online speech without infringing on rights?

    """)

# Data Page
elif page == "Data":
    st.title("Data Collection Process")
    st.subheader("All Code Can Be Found At my Github Repository: [Github-Repo](https://github.com/jordan-kesner/text-mining-data-collection)")

    st.write("""
    The data for this project was collected from multiple sources, including NewsAPI, Reddit API, and web scraping from CNN and Fox News. 
    To ensure balance, queries were carefully designed to retrieve discussions related to both free speech advocacy and content moderation.
    """)

    st.subheader("NewsAPI Data Collection")
    st.write("""
    - **Endpoint:** `https://newsapi.org/v2/everything`
    - **Queries Used:** queries = {
    "Free Speech": [
        "social media free speech",
        "big tech censorship",
        "social media censorship",
        "platform bias in content moderation",
        "First Amendment and social media",
        "government regulation of online speech",
        "Twitter free speech policies",
        "Facebook content restrictions",
        "social media freedom of expression",
        "censorship of political opinions on social media",
        "banned accounts reinstated social media",
        "Section 230 and online speech",
        "is social media limiting free speech",
        "free speech absolutism on social platforms",
        "self-regulation vs government intervention social media",
        "cancel culture and free speech",
        "academic freedom and social media",
        "lawsuits over social media censorship",
        "does deplatforming violate free speech?",
        "corporate influence on free expression"
    ],
    
    "Hate Speech Regulation": [
        "hate speech regulation social media",
        "misinformation and online hate speech",
        "social media content moderation policies",
        "Facebook Twitter content bans",
        "online harassment and platform responsibility",
        "harmful speech regulation online",
        "hate speech laws and digital platforms",
        "big tech misinformation policies",
        "social media extremism policies",
        "AI moderation of hate speech",
        "freedom of speech vs hate speech",
        "hate speech vs political correctness online",
        "social media toxicity prevention",
        "effectiveness of online content moderation",
        "social media policies on hate speech removal",
        "role of AI in detecting hate speech",
        "harm caused by online radicalization",
        "how misinformation spreads online",
        "regulation of political speech on social media",
        "role of fact-checking in content moderation"
    ],
    
    "Neutral Discussion": [
        "impact of content moderation",
        "how do platforms regulate speech",
        "debate over online free speech policies",
        "government vs private sector role in speech regulation",
        "are social media policies fair?",
        "impact of misinformation on democracy",
        "history of free speech regulation",
        "how do social media platforms enforce policies?",
        "effects of AI-driven content moderation",
        "case studies of social media bans",
        "public opinion on online free speech",
        "legal frameworks for online speech",
        "international laws on free speech and hate speech",
        "role of journalists in free speech regulation",
        "what should be considered hate speech?",
        "difference between hate speech and criticism",
        "impact of content moderation on marginalized groups",
        "debate over removing extremist content",
        "effectiveness of different content moderation strategies",
        "how should misinformation be handled?"
    ],
    
    "Mixed Opinions": [
        "is free speech absolute?",
        "can free speech and moderation coexist?",
        "moderation vs censorship",
        "should governments regulate social media?",
        "social media free speech vs misinformation control",
        "can AI fairly moderate online speech?",
        "balancing user safety and free speech",
        "case studies of controversial deplatforming",
        "big tech vs government in free speech control",
        "how to balance hate speech laws with free speech rights",
        "social media companies' responsibility in free speech debates",
        "is there a middle ground for content moderation?",
        "what does fair content moderation look like?",
        "are online bans justified?",
        "do social media companies over-moderate?",
        "how should anonymous speech be regulated?",
        "content moderation case studies from Facebook and Twitter",
        "who decides what qualifies as hate speech?",
        "should controversial figures be deplatformed?",
        "role of human moderators vs AI in content moderation"
    ]
}
    """)

    st.image("newsapi_query.png", caption="Example of NewsAPI Query and Results", width=1000)

    st.subheader("Reddit API Data Collection")
    st.write("""
    - **Subreddits**: "Free Speech": ["freespeech", "Libertarian", "TrueFreeSpeech", "FreeSpeechDebate"],
    "Hate Speech Regulation": ["socialmedia", "technology", "moderation", "InternetSafety", "news"],
    "Neutral Discussion": ["AskReddit", "PoliticalDiscussion", "Technology", "news"],
    "Mixed Opinions": ["changemyview", "TrueNeutral", "ModeratePolitics", "InternetCulture"]
    - **Endpoint:** `reddit.subreddit("subreddit").search(query, limit=50, sort="relevance")`
    - **Queries Used:** "Free Speech": ["social media free speech", "big tech censorship", "online censorship debate"],
    "Hate Speech Regulation": ["hate speech online", "content moderation policies", "social media hate speech regulation"],
    "Neutral Discussion": ["impact of content moderation", "social media free speech study", "how do platforms regulate speech"],
    "Mixed Opinions": ["is free speech absolute?", "can free speech and moderation coexist?", "moderation vs censorship"]
    - **Sorting Applied:** Relevance, Newest Posts
    """)

    st.image("reddit_query.png", caption="Example of Reddit Query and Results", width=1000)

    st.subheader("CNN and Fox News Data Collection")

    st.write("""
    Since CNN and Fox News do not provide free APIs for article retrieval, web scraping techniques were used to collect data.
    The `requests` and `BeautifulSoup` libraries were employed to extract article titles, publication dates, and full text.
    Only publicly available content was scraped, respecting site policies.
    """)
    st.subheader("CNN Web Scraping Code")
    st.image("cnn_scraping_code.png", caption="CNN Web Scraping Code", width=1000)
    st.subheader("CNN Article Example")
    st.image("cnn_article_example.png", caption="CNN Article Example", width=1000)
    st.subheader("Fox News Web Scraping Code")
    st.image("fox_scraping_code.png", caption="Fox News Web Scraping Code", width=1000)
    st.subheader("Fox News Article Example")
    st.image("fox_example_article.png", caption="Fox News Article Example", width=1000)
    st.subheader("CSV and Corpus Files")
    st.image("csv_files.png", caption="Csv Files and Corpus Files", width=1000)
    st.subheader("Raw Dataframes for Sources")
    st.write("""These are the raw dataframes that were created after scraping and saving the data to csvs and corpus files.""")
    st.subheader("NewsAPI Dataframe")
    st.dataframe(newsapi_df_raw)
    st.subheader("Reddit Dataframe")
    st.dataframe(reddit_df_raw)
    st.subheader("CNN Dataframe")
    st.dataframe(cnn_df_raw)
    st.subheader("Fox News Dataframe")
    st.dataframe(foxnews_df_raw)
    st.subheader("WordClouds With Raw Data")
    st.write("""These are the wordclouds created from the raw dataframes.""")
    st.subheader("NewsAPI WordCloud")
    st.image("news_api_raw_cloud.png", caption="NewsAPI WordCloud", width=1000)
    st.subheader("Reddit WordCloud")
    st.image("reddit_raw_cloud.png", caption="Reddit WordCloud", width=1000)
    st.subheader("CNN WordCloud")
    st.image("cnn_raw_cloud.png", caption="CNN WordCloud", width=1000)
    st.subheader("Fox News WordCloud")
    st.image("fox_raw_cloud.png", caption="Fox News WordCloud", width=1000)
    st.subheader("Data Cleaning and Preprocessing")
    st.write("""The data was cleaned and preprocessed using the following steps:
             a. Convert text to lowercase
            b. Get rid of all weird spacing
            c. Remove stopwords
            d. Remove punctuation
            e. Remove numbers
            f. Remove emojis
            g. Define custom stopwords
            h. Remove special characters
            i. Remove two letter words besides “AI”""")
    st.image("cleaning_code.png", caption="Data Preprocessing Steps", width=1000)
    st.subheader("Cleaned Dataframes for Sources")
    st.write("""These are the cleaned dataframes that were created after preprocessing the raw dataframes.""")
    st.subheader("NewsAPI Dataframe")
    st.dataframe(newsapi_cleaned_df)
    st.subheader("Reddit Dataframe")
    st.dataframe(reddit_cleaned_df)
    st.subheader("CNN Dataframe")
    st.dataframe(cnn_cleaned_df)
    st.subheader("Fox News Dataframe")
    st.dataframe(foxnews_cleaned_df)
    st.subheader("WordClouds With Cleaned Data")
    st.write("""These are the wordclouds created from the cleaned dataframes.""")
    st.subheader("NewsAPI CLeaned WordCloud")
    st.image("newsapi_cleaned_cloud.png", caption="NewsAPI WordCloud", width=1000)
    st.subheader("Reddit Cleaned WordCloud")
    st.image("reddit_cleaned_cloud.png", caption="Reddit WordCloud", width=1000)
    st.subheader("CNN Cleaned WordCloud")
    st.image("cnn_cleaned_cloud.png", caption="CNN WordCloud", width=1000)
    st.subheader("Fox News Cleaned WordCloud")
    st.image("foxnews_cleaned_cloud.png", caption="Fox News WordCloud", width=1000)
    st.subheader("NewsAPI Text Transformation Vectorization Examples")
    st.write("""The cleaned dataframes were transformed into various forms. They were made into dataframes for CountVectorizer,
             TfidfVectorizer, Stemming, and Lemmatization. The dataframes were then saved to csv files.
             Below are examples of each type of transformation.""")
    st.subheader("CountVectorizer Dataframe")
    st.dataframe(newsapi_countvectorizer_df)
    st.subheader("TfidfVectorizer Dataframe")
    st.dataframe(newsapi_tfidfvectorizer_df)
    st.subheader("Stemmed Dataframe")
    st.dataframe(newsapi_stemmed_df)
    st.subheader("Lemmatized Dataframe")
    st.dataframe(newsapi_lemmatized_df)
    st.subheader("Merged Cleaned Dataframe")
    st.dataframe(merged_df)






elif page == "About Me":
    st.title("About Me")
    st.write("""Hi! My name is Jordan and I'm an Information Science Master's student at the University of Colorado Boulder.
              I'm passionate about data science and machine learning techniques, and I'm excited to share my work with you!""")

else if page == "Clustering":
    st.title("Clustering")
    st.write("""In this section, we will explore the clustering of the data using KMeans.""")
