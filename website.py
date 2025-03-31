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
newsapi_df_raw = load_data("raw_dataframes/newsapi_raw_df.csv")
reddit_df_raw = load_data("raw_dataframes/reddit_raw_df.csv")
cnn_df_raw = load_data("raw_dataframes/cnn_raw_df.csv")
foxnews_df_raw = load_data("raw_dataframes/foxnews_raw_df.csv")
newsapi_cleaned_df = load_data("cleaned_dataframes/newsapi_cleaned_df.csv")
reddit_cleaned_df = load_data("cleaned_dataframes/reddit_cleaned_df.csv")
cnn_cleaned_df = load_data("cleaned_dataframes/cnn_cleaned_df.csv")
foxnews_cleaned_df = load_data("cleaned_dataframes/foxnews_cleaned_df.csv")
newsapi_countvectorizer_df = load_data("countvectorized_dataframes/NewsAPI_count_vectorized.csv")
newsapi_tfidfvectorizer_df = load_data("tfidf_dataframes/NewsAPI_tfidf_vectorized.csv")
newsapi_stemmed_df = load_data("stemmed_dataframes/NewsAPI_stemmed.csv")
newsapi_lemmatized_df = load_data("lemmatized_dataframes/NewsAPI_lemmatized.csv")
merged_df = load_data("cleaned_dataframes/cleaned_merged_df.csv")



# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data","About Me","Clustering","Association Rule Mining","LDA"])

# Introduction Page
if page == "Introduction":
    st.title("Analyzing Online Discourse: Free Speech vs. Hate Speech Regulation")
    st.write("""
Social media platforms have transformed how people communicate, making them central to public discourse. While these platforms provide unprecedented opportunities for expression, they also raise concerns about the spread of harmful content. Supporters of absolute free speech argue that online platforms should allow all viewpoints, emphasizing that limiting speech on private platforms undermines democratic principles. However, proponents of content moderation highlight the dangers of misinformation, hate speech, and harassment, arguing that unchecked speech can lead to real-world harm. The tension between free expression and platform regulation has led to controversies over censorship, misinformation policies, and political bias in moderation decisions. High-profile cases, such as the banning of former U.S. President Donald Trump from Twitter and Facebook, have intensified debates about whether companies should control speech on their platforms. Some view content moderation as a necessary safeguard against extremism and misinformation, while others see it as an overreach of corporate and government power. The role of artificial intelligence in moderating content further complicates the issue, as AI models can misinterpret context, flagging satire or political speech as harmful. Global differences in free speech laws also impact how companies enforce moderation policies, creating inconsistencies across platforms. As digital spaces become the primary arenas for political and cultural debates, the question of how to balance free speech with responsible moderation remains highly contested.

Defining the boundaries of free speech online presents significant legal, ethical, and technological challenges. In the United States, the First Amendment protects most forms of speech from government censorship, but private companies are not legally required to uphold these protections. In contrast, countries such as Germany and France enforce strict hate speech laws, requiring platforms to remove harmful content within hours or face penalties. As governments explore potential regulations, some advocate for reforming Section 230, a U.S. law that shields social media companies from liability for user-generated content. Meanwhile, alternative platforms, such as Truth Social and Parler, have emerged as spaces promoting unrestricted speech, raising questions about how decentralized media environments shape public discourse. The increasing reliance on AI-driven moderation systems also introduces new concerns about bias, accuracy, and transparency in decision-making. While AI can process vast amounts of content at scale, human intervention is often required to ensure fairness in enforcement. The extent to which content moderation influences elections, social movements, and political polarization remains an open question. As digital platforms continue to evolve, determining whether free speech and content moderation can coexist—or whether one will inevitably restrict the other—remains a critical issue for governments, corporations, and society as a whole.

The way people discuss free speech and moderation often reveals deeper beliefs about rights, power, and social responsibility. The language used when debating these topics, such as referring to moderation as either censorship or protection, can shape how online communities and beyond understand these issues. By analyzing patterns in word usage and topic associations, it becomes possible to trace the strategies people use to justify their views and positions on this complex issue. Beyond legal concerns, the debate over online speech is deeply shaped by internet culture and public opinion. Memes, hashtags, and viral posts ofte influence how policies are percieved. As a result, online speech debates reflect not only legal principles but also emotional responses, ideological identities, and shifting norms around what speech is acceptable in public, digital spaces.
             
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
    st.image("images/censorship.jpg")
    st.image("images/say_no_to_hate.png")

# Data Page
elif page == "Data":
    st.title("Data Collection Process")
    st.subheader("Data Collection code can be found on my GitHub Repo in the notebook \"Text_Mining_Project_Data_Collection.ipynb\": [Github-Repo](https://github.com/jordan-kesner/text-mining-social-media-project)")

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

    st.image("images/newsapi_query.png", caption="Example of NewsAPI Query and Results", width=1000)

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

    st.image("images/reddit_query.png", caption="Example of Reddit Query and Results", width=1000)

    st.subheader("CNN and Fox News Data Collection")

    st.write("""
    Since CNN and Fox News do not provide free APIs for article retrieval, web scraping techniques were used to collect data.
    The `requests` and `BeautifulSoup` libraries were employed to extract article titles, publication dates, and full text.
    Only publicly available content was scraped, respecting site policies.
    """)
    st.subheader("CNN Web Scraping Code")
    st.image("images/cnn_scraping_code.png", caption="CNN Web Scraping Code", width=1000)
    st.subheader("CNN Article Example")
    st.image("images/cnn_article_example.png", caption="CNN Article Example", width=1000)
    st.subheader("Fox News Web Scraping Code")
    st.image("images/fox_scraping_code.png", caption="Fox News Web Scraping Code", width=1000)
    st.subheader("Fox News Article Example")
    st.image("images/fox_example_article.png", caption="Fox News Article Example", width=1000)
    st.subheader("CSV and Corpus Files")
    st.image("images/csv_files.png", caption="Csv Files and Corpus Files", width=1000)
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
    st.image("images/news_api_raw_cloud.png", caption="NewsAPI WordCloud", width=1000)
    st.subheader("Reddit WordCloud")
    st.image("images/reddit_raw_cloud.png", caption="Reddit WordCloud", width=1000)
    st.subheader("CNN WordCloud")
    st.image("images/cnn_raw_cloud.png", caption="CNN WordCloud", width=1000)
    st.subheader("Fox News WordCloud")
    st.image("images/fox_raw_cloud.png", caption="Fox News WordCloud", width=1000)
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
    st.image("images/cleaning_code.png", caption="Data Preprocessing Steps", width=1000)
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
    st.image("images/newsapi_cleaned_cloud.png", caption="NewsAPI WordCloud", width=1000)
    st.subheader("Reddit Cleaned WordCloud")
    st.image("images/reddit_cleaned_cloud.png", caption="Reddit WordCloud", width=1000)
    st.subheader("CNN Cleaned WordCloud")
    st.image("images/cnn_cleaned_cloud.png", caption="CNN WordCloud", width=1000)
    st.subheader("Fox News Cleaned WordCloud")
    st.image("images/foxnews_cleaned_cloud.png", caption="Fox News WordCloud", width=1000)
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

elif page == "Clustering":
    st.title("Clustering")
    st.subheader("The code for clustering can be found at my [Github-Repo](https://github.com/jordan-kesner/text-mining-social-media-project), K-means inside the notebook \"Text_Mining_Project_Clustering.ipynb\" and Hierarchical Clustering in R inside the notebook \"hclustering.Rproj\"")

    st.header("What is Clustering?")
    st.markdown("""
Clustering is an unsupervised machine learning technique that groups similar data points together without predefined labels. Unlike classification, which relies on labeled data, clustering allows us to explore underlying structures in a dataset and discover natural groupings. In the context of text analysis, clustering helps identify patterns in textual data, revealing how different discussions relate to one another.

For this project, we implement two clustering methods:
- **K-Means Clustering**: A popular centroid-based method that partitions data into k groups, assigning each data point to the nearest cluster center.
- **Hierarchical Clustering (HClust)**: A technique that builds a tree-like hierarchy of clusters, allowing us to visualize how documents relate at different levels of similarity.

By applying these clustering techniques to our dataset, we can identify distinct discussion themes and analyze how free speech and hate speech regulation debates are structured.
    """)

    st.header("Why Use Clustering in this Project?")

    st.markdown("""
    The goal of this project is to analyze discussions on free speech and hate speech regulation using text data from multiple sources. Since these discussions can take various forms—supportive, critical, neutral, or mixed—clustering allows us to:

    - Discover hidden patterns in how different viewpoints are expressed.
    - Identify distinct discussion groups without relying on pre-existing labels.
    - Validate or refine dataset labels (e.g., do clusters align with "free speech" and "hate speech regulation" categories, or do new themes emerge?).

    By clustering articles and posts, we aim to gain insights into whether these discussions naturally divide along ideological lines or if certain themes dominate across all sources.
    """)
    st.header("Distance Metrics Used")
    st.markdown("""Since clustering is based on similarity, the choice of distance metric plays a crucial role in how clusters are formed.

    1. Euclidean Distance (for K-Means)

    - Measures the straight-line distance between two points in space.
    - Works well when clusters have spherical shapes and are well-separated.
    - Less effective for high-dimensional text data, but still widely used.
                
    2. Cosine Similarity (for Hierarchical Clustering)

    - Measures the angle between two text vectors instead of their absolute distance.
    - More effective for sparse and high-dimensional datasets like TF-IDF text representations.
    - Helps group texts that share similar topics and language usage, even if they differ in length.
                
    By using Euclidean distance for K-Means and Cosine Similarity for Hierarchical Clustering, we can evaluate how well each method captures the relationships between discussions.
    """)
    st.header("Data Preparation")
    st.markdown("""Clustering only works on numerical data, so we need to transform our text data into a format that algorithms can process. In this project, we use two common text vectorization techniques:
                - CountVectorizer: Converts text into a matrix of word counts.
                - TF-IDF Vectorizer: Assigns weights to words based on their importance in documents.
                Here you can see an example of the data before and after vectorization:""")
    st.subheader("Cleaned Dataframe")
    st.image("images/cleaned_df.png", caption="Cleaned Dataframe", width=1000)
    st.subheader("TF-IDF Vectorizer Dataframe")
    st.image("images/tfidf_df.png", caption="TF-IDF Vectorizer Dataframe", width=1000)
    st.header("Clustering Results")
    st.subheader("K-Means Clustering")
    st.markdown("""I tested K-Means clustering with k values of 3,4, and 5.
                 Using the Silhouette method, the highest average score occurred at k = 4, suggesting this as the optimal number of clusters.
                 The Elbow Method also showed a noticeable inflection point at k = 4.""")
    st.image("images/silhouette_kmeans.png", caption="K-Means Silhouette Scores", width=1000)
    st.image("images/elbow_method_kmeans.png", caption="K-Means Elbow Method", width=1000)
    st.markdown("""After reducing the dimensionality of the TF-IDF vectors using PCA, I visualized the K-Means clustering in 2D/3D space. The plot revealed moderately distinct clusters with some overlap, indicating variations in topic boundaries across documents.
    """)
    st.image("images/kmeans_viz.png", caption="K-Means Clustering (2D)", width=1000)
    st.image("images/k_means_3d.png", caption="K-Means Clustering (3D)", width=1000)
    st.markdown("""Top words in each cluster helped identify the major themes represented in the dataset.

    - Cluster 0 appeared to center around technology marketing and platform launches, with dominant terms like “launches,” “google,” “tiktok,” “ads,” “meta,” and “instagram.” This suggests the cluster captured discussions about product updates, advertising, and features from major tech platforms.

    - Cluster 1 strongly reflected free speech advocacy and government involvement, with words such as “speech,” “free speech,” “government,” “censorship,” “media,” and “right.” This group most closely aligned with traditional free speech discourse, particularly in relation to media companies and political rights.

    - Cluster 2 was more conversational and opinion-driven, characterized by words like “trump,” “think,” “political,” “right,” and “want.” This cluster likely captured user commentary and informal political discussion, often seen in platforms like Reddit.

    - Cluster 3 revolved around social media platforms and content moderation, with frequent terms like “media,” “social media,” “content,” “tiktok,” “users,” and “platforms.” This indicates that the cluster focused on debates about how content is managed, user engagement, and the role of platforms in shaping discourse.

    These clusters reveal a meaningful separation of news vs. opinion, platform functionality vs. regulation, and formal free speech debates vs. social media discourse, all of which are highly relevant to the project’s focus on speech regulation in digital spaces.


                
    """)

    st.markdown("""To assess how well the unsupervised K-Means clustering aligned with the manually assigned labels, I compared the cluster assignments to the original categories: Free Speech, Hate Speech Regulation, Mixed Opinions, and Neutral Discussion. The results showed a mix of clear alignments and overlaps:

    - Cluster 0 was highly specific, containing only documents labeled as Hate Speech Regulation (34 entries). This suggests that this cluster captured a relatively focused and coherent topic area, likely centered on policy discussions, platform regulation, and moderation practices.

    - Cluster 1 leaned heavily toward Free Speech (116 documents), but also included a notable number of Mixed Opinions (45) and Neutral Discussions (25). This cluster likely captured broader or more ideological conversations about speech rights that sometimes overlapped with general commentary or debate.

    - Cluster 2 was the most heterogeneous, with a large concentration of Mixed Opinions (144), as well as a near-even mix of Free Speech (40) and Hate Speech Regulation (38). This indicates the cluster reflects ambiguous or blended perspectives, and may represent posts where the framing of speech issues was less clearly aligned with a single viewpoint.

    - Cluster 3 was dominated by Hate Speech Regulation (155), but also included a mix of Free Speech (47) and Neutral Discussion (75), suggesting this cluster captured content that intersected between policy enforcement, user rights, and broader platform debates.

    These results show that while some clusters (like Cluster 0) captured highly specific themes, others revealed the complex, overlapping nature of real-world speech debates, particularly where mixed or nuanced opinions are common. Overall, clustering provided useful insight into the thematic structure of the data, but also highlighted that speech-related discussions do not always fall neatly into one label, which supports the use of unsupervised methods for exploratory analysis.
                    """)
    st.image("images/kmeans_cluster_comparison.png", caption="K-Means Cluster Comparison", width=1000)
    st.subheader("Hierarchical Clustering")
    st.markdown("Hierarchical Clustering (HClust) is a tree-like clustering method that groups documents based on their similarity. By constructing a dendrogram, we can visualize how documents are related at different levels of granularity, revealing both broad and fine-grained clusters.")
    st.markdown("""I created a dendrogram using the Ward method, which minimizes the variance between clusters. The dendrogram showed distinct clusters at different heights, indicating how documents group together based on their content. By cutting the dendrogram at different heights, we can explore clusters at varying levels of similarity, from broad themes to more specific topics. After using the Silhouette method to evaluate the optimal k value, I found that k = 7 a good score, but not the best score as you can see in the plot below. I chose 7 in order to keep the number of clusters smaller and more manageable for this dataset.""")
    st.image("images/silhouette_hclust.png", caption="Hierarchical Clustering Silhouette Scores", width=1000)
    st.markdown("""The dendrogram revealed several key clusters:""")
    st.image("images/dendrogram.png", caption="Hierarchical Clustering Dendrogram", width=1000)
    st.markdown("""Hierarchical clustering revealed a variety of distinct themes across the dataset, with each cluster grouping together related types of discussion.
            
                - Cluster 1 captured more casual or conversational language. Words like “think,” “say,” “want,” and “really” suggest this group includes opinion-based posts or general reflections—likely content that didn’t take a firm stance but still engaged with the topic.
                - Cluster 2 focused heavily on TikTok and Chinese government involvement. Terms like “china,” “ban,” “trump,” and “law” suggest this cluster reflects geopolitical debates, particularly around platform bans and national security concerns.
                - Cluster 3 blended themes of platform regulation and political speech. With words like “trump,” “speech,” “meta,” and “companies,” this cluster seemed to sit at the intersection of tech platform decisions and public political figures—especially in relation to content moderation.
                - Cluster 4 stood out as very Reddit-specific. Top words like “subreddit,” “mod,” “community,” and “banned” made it clear this group was centered around moderation practices, user rules, and community governance on Reddit.
                - Cluster 5 was the most clearly aligned with free speech advocacy. It featured words like “free speech,” “censorship,” “government,” and “freedom,” pointing to debates about individual rights, state overreach, and platform responsibility.
                - Cluster 6 leaned toward social media engagement content. With words like “post,” “follow,” “instagram,” and “questions,” it seemed to reflect how users interact with platforms, rather than discussing policy or regulation directly.
                - Cluster 7 focused on tech product launches and marketing, with terms like “launches,” “google,” “youtube,” and “ads.” This cluster pulled in content about updates, features, and platform branding—less about speech and more about the companies themselves.
                    
        
                """)
    st.markdown("Overall, the hierarchical approach helped surface more specific and structured groupings, especially those tied to particular platforms or subtopics. Compared to K-Means, HClust gave a more detailed picture of how online speech discussions split—not just between free speech and regulation, but across tech, politics, user behavior, and platform identity.")
    st.markdown("""To evaluate the alignment between the hierarchical clusters and the original labels, I compared the cluster assignments to the manually assigned categories: Free Speech, Hate Speech Regulation, Mixed Opinions, and Neutral Discussion. """)
    st.image("images/hclust_compare_with_labels.png", caption="Hierarchical Clustering Comparison with Labels", width=1000)
    st.markdown("""
                When comparing the hierarchical clusters to the original labels, some clear patterns emerged. Cluster 5 was mostly aligned with Free Speech content, containing 84 such documents and very few from other categories. Cluster 6 and Cluster 7 were heavily skewed toward Hate Speech Regulation, suggesting those clusters captured more focused policy or regulation-oriented discussions. On the other hand, Cluster 1 was the most mixed, with a fairly even spread across all labels—particularly strong in Mixed Opinions (106 documents) and Hate Speech Regulation (62). This suggests it grouped more general or ambiguous discussions. Cluster 3 also showed a broad spread, reflecting overlap between speech rights, regulation, and neutral commentary. Overall, HClust revealed both clear topic-based clusters and clusters that reflected the complexity of real-world conversations, where different viewpoints often blend within the same thematic space.""")
    st.subheader("Comparing K-Means and Hierarchical Clustering")
    st.markdown("""
                Although K-Means and Hierarchical Clustering (HClust) used different approaches and produced different numbers of clusters, there was significant overlap between their groupings.
                For example, HClust Cluster 1 was split mostly between K-Means Cluster 2 (132 documents) and K-Means Cluster 3 (108 documents), suggesting that both methods grouped these documents as part of the same general topic area—likely one of mixed or overlapping opinions.
                HClust Cluster 5, which strongly aligned with Free Speech content based on top words, was concentrated in K-Means Cluster 1 (113 documents). This indicates strong agreement between the methods on that thematic group.
                Some clusters showed less alignment. For instance, K-Means Cluster 0, which contained only Hate Speech Regulation documents, didn't clearly map to any one HClust cluster—it only appeared in Cluster 6 (2 docs) and Cluster 7 (32 docs), suggesting that HClust divided that topic into finer subgroups.
                Interestingly, HClust Cluster 7 was completely isolated in K-Means Cluster 0, suggesting a one-to-one mapping and a very distinct group of documents, likely reflecting a narrow topic like platform launches or marketing.
                Overall, K-Means created broader clusters with mixed content (e.g., Clusters 2 and 3), while HClust offered more granular separation, breaking those large groups into smaller, more topic-specific clusters. The methods generally agreed on the core themes but approached them with different levels of specificity.
                """)
    st.subheader("Conclusion")
    st.markdown(""" Clustering the text data revealed interesting patterns in the discussion of free speech and hate speech censorship.
                Through K-Means and Hierarchical Clustering, I was able to find distinct themes and groupings, like discussions centered around government censorship, platform moderation, tech products, and informal public opinion.
                One key takeaway is that these conversations don't always fall into neat categories. Some clusters aligned closely with the predefined labels like Free Speech or Hate Speech Regulation, others captured more nuanced or overlapping opions, especially those labeled Mixed Opinions or Neutral Discussion. This suggests that speech debates are complex and multifaceted, with a wide range of perspectives and viewpoints. These real-world discourses on controversial topics often blend together, making perspectives blend rather than divide cleanly.
                Hierarchical Clustering, in particular, helped reveal more specific subtopics, like Reddit moderation or TikTok bans, while K-Means was more effective at identifying broad thematic divisions. Together, both methods demonstrated that unsupervised clustering can be a powerful tool for exploring and interpreting large-scale discussions on social and political issues.
                Ultimately, clustering helped me better understand the structure and diversity of speech-related conversations across media platforms, and showed how machine learning can assist in uncovering themes that may not be obvious from manual labeling alone.
                """)

elif page == "Association Rule Mining":
    st.title("Association Rule Mining")
    st.subheader("What is Association Rule Mining?")
    st.markdown("Association Rule Mining (ARM) is a fundamental data mining technique used to discover interesting relationships, correlations, or patterns among items in large datasets. ARM is commonly used on transactional data, such as market basket analysis, where the goal is to identify items that are frequently purchased together. In the context of text analysis, ARM can be applied to identify co-occurring terms, phrases, or topics within documents, revealing hidden connections and associations. ARM aims to uncover rules like 'if a customer buys product A, they are likely to buy product B' or 'if a document mentions term X, it is likely to mention term Y'. To find these rules, ARM uses three metrics:")
    st.markdown("Support measures how frequently an itemset appears in the dataset. It is calculated as the number of transactions containing the itemset divided by the total number of transactions.")
    st.markdown("Confidence measures the likelihood that an item B is purchased when item A is purchased. It is calculated as the number of transactions containing both items A and B divided by the number of transactions containing item A.")
    st.markdown("Lift compares the likelihood of buying items A and B together to the likelihood of buying them independently. It is calculated as the confidence of the rule divided by the support of item B.")
    st.image("images/AR_1.png", caption="Association Rule Mining Metrics", width=1000)
    st.markdown("To extract these rules, the Apriori algorithm is commonly used. It identifies frequent itemsets by iteratively pruning infrequent items and generating candidate itemsets based on the support threshold. The algorithm then generates association rules from the frequent itemsets, calculating confidence and lift to evaluate rule strength.")
    st.subheader("Why Use Association Rule Mining in this Project?")
    st.markdown("In this project, ARM will be used on my text data to identify co-occurring terms, phrases, or topics within documents. By applying ARM, I aim to uncover hidden relationships between words or concepts in the discussions on free speech and hate speech regulation. This will help reveal patterns in how different terms are used together, potentially highlighting common themes, topics, or viewpoints. ARM can provide valuable insights into the structure of the text data, revealing underlying connections that may not be immediately apparent from manual analysis. By extracting association rules, I can better understand how different terms relate to one another and how they collectively shape the discourse on speech regulation.")
    st.subheader("Association Rule Mining Data Preparation")
    st.markdown("Before applying ARM, the text data needs to be preprocessed and transformed into a suitable format. I will use the cleaned dataframes and remove the labels to prepare the data for ARM. The data will be converted into a transactional format, where each document is treated as a transaction containing the terms present in that document. This format allows the Apriori algorithm to identify frequent itemsets and generate association rules based on term co-occurrence.")
    st.markdown("The raw data before converting into transactions.")
    st.image("images/cleaned_df_raw.png", caption="Cleaned Raw Data", width=1000)
    st.markdown("The data after converting into transactions.")
    st.image("images/transactions.png", caption="Transactions Data", width=1000)
    st.subheader("Association Rule Mining Results")
    st.markdown("After applying the Apriori algorithm to the text data, I will extract association rules based on the support, confidence, and lift metrics. These rules will reveal interesting relationships between terms, phrases, or topics in the discussions on free speech and hate speech regulation. By analyzing the association rules, I can identify common patterns, co-occurring concepts, or related themes that emerge from the text data. This will provide valuable insights into how different terms are connected and how they collectively shape the discourse on speech regulation.")
    st.markdown("After converting my data into transactions, I tried to do ARM on the full dataset, which had 28745 unique words, and 826 transactions. This huge item space and small number of transactions made it difficult to run the Apriori algorithm and R hit a memory limit trying to store all candidate itemsets. To fix this issue I rerun ARM on a smaller dataset with only the top 5700 unique words.")
    st.markdown("The following parameters were used for the Apriori algorithm:")
    st.markdown("Support: 0.015")
    st.markdown("Confidence: 0.4")
    st.markdown("Maximum Rule Length: 2")
    st.markdown("The top rules by Support were:")
    st.image("images/top_15_rules_support.png", caption="Top 15 Rules by Support", width=1000)
    st.markdown("The rules with the highest support reflected the most frequent word co-occurences in the dataset. Many high-support rules included only single common terms and did not offer much analytical depth.")
    st.markdown("The top rules by Confidence were:")
    st.image("images/top_15_rules_confidence.png", caption="Top 15 Rules by Confidence", width=1000)
    st.markdown("The high-confidence rules were more revealing, showing strong relationships between terms that frequently appeared together in the dataset. These rules indicated which terms were likely to co-occur based on the presence of a single term.")
    st.markdown("Examples include:")
    st.markdown("{boycott} -> {speech}: indicating that discussions involving boycotts often mentioned speech, possibly refencing platform bans or content removals.")
    st.markdown("{absolutist} -> {speech} and {absolutist} -> {free}: suggesting that users are discussing free speech absolutism, a well-known ideological position on the topic.")
    st.markdown("{overreach} => {government}: revealing how users frame moderation or censorship as government interference.")
    st.markdown("{inflammatory} => {think}: which suggests reflections on what constitutes “inflammatory” content.")
    st.markdown("The top rules by Lift were:")
    st.image("images/top_15_rules_lift.png", caption="Top 15 Rules by Lift", width=1000)
    st.markdown("High-lift rules show strong, non-random relationships between words or items.")
    st.markdown("Examples include:")
    st.markdown("{slippery} => {slope} and {crowded} => {theater}: reflect classic free speech metaphors — the slippery slope argument and the “fire in a crowded theater” legal analogy.")
    st.markdown("{hunter} => {laptop}: alludes to political controversies surrounding censorship and media coverage.")
    st.markdown("{rulebreaking} => {slogans} and {faq} => /{summary/} may reflect content structuring or moderation guidelines.")
    st.markdown("Visualizations of ARM")
    st.image("images/ARM_plot.png", caption="Association Rule Mining Plot", width=1000)
    st.markdown("This plot shows the top 15 rules by lift. The size of the nodes represents the support of the rule, while the color represents the confidence. The thickness of the edges indicates the lift value, with thicker edges indicating higher lift.")
    st.image("images/matrix_plot_R.png", caption="Association Rule Mining Matrix Plot", width=1000)
    st.subheader("Conclusion")
    st.markdown("Through association rule mining, I uncovered common patterns in how people talk about free speech, moderation, and online content. The results revealed that users often connect ideas like “government overreach” with censorship, or describe free speech in absolutist terms, suggesting strong opinions about where the line should be drawn. Some posts referenced famous legal metaphors, like “slippery slope” or “crowded theater,” showing how users frame their arguments using widely recognized rhetorical tools. I also saw associations that reflected political flashpoints, like Hunter Biden’s laptop, which often comes up in debates about media bias and content removal. These patterns help highlight not only what people are saying, but how they build their arguments—often blending ideology, metaphor, and political context—when discussing hate speech and free expression online.")

elif page == "LDA":
    st.title("Latent Dirichlet Allocation (LDA)")
    st.subheader("What is Latent Dirichlet Allocation (LDA)?")
    st.markdown("Before getting into LDA, what is topic modeling? Topic modeling is a type of statistical model used for discovering the abstract \"topics\" that occur in a collection of documents. It is a powerful tool for exploring large datasets of text data and classifying text in a document to a topic.")
    st.markdown("Latent Dirichlet Allocation (LDA) is a popular topic modeling technique and is used to classify text in a document to a particular topic. It assumes that each document is a mixture of topics and that each word in the document is attributable to one of the document's topics. LDA is based on the idea that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words.")
    st.markdown("LDA is considered an unsupervised learning method because it does not require labeled data. It automatically identifies topics in a text corpus by analyzing the co-occurrence patterns of words in the documents.")
    st.subheader("Why Use LDA in this Project?")
    st.markdown("I am using LDA in this project to uncover topics in the discussions surrounding free speech and hate speech regulation. By applying LDA to the text data, I aim to identify the underlying themes and topics that emerge from the documents. This will help reveal the key subjects of conversation, the main issues being discussed, and the different perspectives present in the dataset. LDA will provide a structured way to explore the content of the documents and understand the distribution of topics across the dataset.")
    st.subheader("LDA Data Preparation")
    st.markdown("Before using LDA, I need to format my data in a way that LDA can use to perform topic modeling. I will use my cleaned dataframe and transform the text using CountVectorizer so that there are word counts for each word in each document. This will create a document-term matrix that LDA can use to identify topics.")
    st.markdown("The raw data before converting into a document-term matrix.")
    st.image("images/cleaned_df_raw.png", caption="Cleaned Raw Data", width=1000)
    st.markdown("The data after converting using CountVectorizer.")
    st.image("images/countvect_badwords.png", caption="CountVectorizer Data With Not Important Words", width=1000)
    st.markdown("When first creating a countvectorizer dataframe of my data, I included all words in the dataset besides stop words, but some words that lack any meaningful information still slipped through such as \"aaa\" and \"aaib\". To fix this issue, I recreated my countvectorizer dataframe removing words that are junk.")
    st.markdown("The data after removing junk words.")
    st.image("images/countvect_good_words.png", caption="CountVectorizer Data Without Junk Words", width=1000)
    st.markdown("Now we can see that only real and important words are being used for LDA.")
    st.subheader("LDA Results")
    st.markdown("I chose to use 6 topics for LDA, so the method will try to find 6 topics in the dataset.")
    st.markdown("""

    To identify common themes across Reddit discussions of online speech, I applied Latent Dirichlet Allocation (LDA) to extract latent topics from the cleaned post content. The model was set to uncover six topics, each representing a cluster of frequently co-occurring words that suggest how users frame and approach the subject of free speech, hate speech, and online platforms.

    ---

    **Topic 1 – Government, Law, and Free Speech**  
    This topic centers on legal and political framing of speech, with words like "government," "amendment," "court," "law," "content," and "censorship." Users in this cluster appear focused on constitutional rights, the role of tech companies, and debates around state involvement in regulating online expression.

    **Topic 2 – Research, Policy, and Public Data**  
    With terms like "data," "study," "state," "research," "health," and "control," this topic reflects a more analytical or research-based angle, possibly discussing how speech and platform behaviors are being studied in academic or policy contexts.

    **Topic 3 – Personal Beliefs and Free Expression**  
    This topic includes subjective and reflective language such as "think," "say," "want," "make," "really," "believe," and "actually." Combined with "speech," "free," and "social," this cluster likely captures everyday perspectives, personal beliefs, and emotional appeals in speech debates.

    **Topic 4 – AI, Technology, and Platforms**  
    Centered around "openai," "chatgpt," "model," "users," "search," and "voice," this topic reflects conversations about emerging tech and AI moderation. It suggests growing public interest in how AI is shaping the future of content creation and regulation.

    **Topic 5 – Platforms and Community Engagement**  
    With terms like "tiktok," "reddit," "instagram," "post," "ads," "youtube," and "community," this topic focuses on social media platforms, user-generated content, and community dynamics. It highlights the ecosystem in which moderation and speech policies are applied.

    **Topic 6 – Global Politics and Power**  
    This topic brings in geopolitical language like "china," "american," "party," "power," "state," and "support," often paired with "trump," "government," and "right." It reflects discussions that place online speech in the context of national identity, global conflict, or political movements.

    ---

    The first visualization displays the top 30 words in one topic, highlighting the depth of terms related to legal rights and censorship. The second visualization presents the top 10 words for each of the six topics, illustrating how each topic captures a different dimension of the public conversation—from law to tech to personal opinion.

    Together, these topics reveal that users frame online speech in diverse and layered ways: some focus on constitutional freedoms and government roles, others emphasize personal views or platform experiences, a growing portion engage with AI and moderation technologies, and some frame the issue through a lens of politics, power, and identity.

    This suggests that conversations about hate speech and free expression are not only widespread, but also deeply connected to broader social, technological, and political themes.
    """)
    st.image("images/LDA_topic_1.png", caption="LDA Topic 1", width=1000)
    st.image("images/LDA_topics.png", caption="LDA Topics", width=1000)
    st.subheader("Conclusion")
    st.markdown("""
The topic modeling revealed that people discuss free speech and content moderation from many different angles. Some users focus on government policies and legal rights, while others share personal beliefs or react to specific platforms like Reddit and TikTok. There was also a clear interest in how new technologies like AI are shaping online conversations. Overall, the topics uncovered by the model show that the debate around speech online is complex and influenced by politics, technology, and individual perspectives.
""")
    


    


