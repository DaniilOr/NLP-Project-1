import datetime
from courlan import check_url
import frontend.utils as tp
import streamlit as st
import numpy as np
import pandas as pd
import torch, requests, time
import numpy as np
import plotly.express as px
from frontend.cfg import ROOT
from streamlit_searchbox import st_searchbox
from thefuzz import process, fuzz

VALID_SRC = False


def search(searchterm: str, ):
    base_urls = tp.get_base_urls()
    return [i[0] for i in process.extract(searchterm, base_urls + [searchterm], scorer=fuzz.ratio)[:5]]


def valid_url(news_src: str = "https://www.bbc.com/news"):
    """
    Check the validity of the News URL entered.
    
    Args:
        news_src (str): The URL of the webpage to analyse.
    """
    global VALID_SRC
    print(news_src)
    try:
        request = requests.get(news_src)
        if request.status_code != 200:
            print("Cannot connect", request.text)
            VALID_SRC = False
        else:
            VALID_SRC = True
    except:
        print("Failed to connect")
        VALID_SRC = False

    return


def plot_fact_bias(biasfig, factfig, source, date):
    """
    Plots the bar and the pie charts on the webpage.
    """
    st.markdown(f"<h3 style='text-align: center; color: black;'>{source} (updated at {date})</h3>",
                unsafe_allow_html=True)
    source_base = check_url(source)
    if source_base:
        source_base = source_base[1]
        tags = tp.get_tags_by_source(source_base)
        if tags:
            st.markdown(f"<p>This site is: <strong>{tags[0]}</strong></p>",
                        unsafe_allow_html=True
                        )
        gpt_responses = tp.get_gpt(source_base)
        if gpt_responses:
            st.write(
                "This source has the following opinion on different topics",
                pd.DataFrame.from_dict(gpt_responses,
                                       orient='index',
                                       columns=['opinion']).T
            )

    st.write("Bias Scores")
    st.plotly_chart(biasfig, use_container_width=True)
    st.write("\nFactuality Results")
    st.plotly_chart(factfig, use_container_width=True)


def plot_ident_pers(identfig, persfig):
    st.write("Identity Framing Results")
    st.plotly_chart(identfig, use_container_width=True)
    st.write("\nPersuasion Results")
    st.plotly_chart(persfig, use_container_width=True)


def plot_results(results):
    results = tp.aggr_scores(results)

    # print('\n\n\n', results)
    t = pd.DataFrame.from_dict(results['nela'],
                               orient='index').T
    t.columns = ['Lexical Diversity',
                 'Average word length',
                 'Average wordcount',
                 'Flesch-Kincaid Readability',
                 'SMOG Grade Readability',
                 'Coleman–Liau index',
                 'LIX',
                 'Moral Foundation: Care',
                 'Moral Foundation: Harm',
                 'Moral Foundation: Fairness',
                 'Moral Foundation: Cheating',
                 'Moral Foundation: Loyalty',
                 'Moral Foundation: Betrayal',
                 'Moral Foundation: Authority',
                 'Moral Foundation: Subversion',
                 'Moral Foundation: Purity',
                 'Moral Foundation: Degradation',
                 'General Moral Foundation'
                 ]


    biasfig = tp.plotbias(results['bias_results'])
    factfig = tp.plotfact(results['factuality_results'])

    print(results['bias_results'], results['factuality_results'])

    identity_results, persuasion_results = tp.get_parq(news_src = news_src)
    is_identity_persuasion = True if identity_results else False

    if not is_identity_persuasion:
        st.write("Identity Framing and Persuasion Results were not found in the database. Displaying Factuality and Bias Results only.")
        plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
    else:
        identfig = tp.plotiden(identity_results)
        persfig = tp.plotpers(persuasion_results)

        plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
        plot_ident_pers(identfig, persfig)
    st.markdown(
        fr"<h4>The Lexical diversity of the text is {t['Lexical Diversity'].iloc[0]:.3f}<br>Average word length: {t['Average word length'].iloc[0]:.3f}<br>Average wordcount: {t['Average wordcount'].iloc[0]:.3f}<br><br>The readability scores are shown below</h4>",
        unsafe_allow_html=True)
    st.write(t[['Flesch-Kincaid Readability',
                'SMOG Grade Readability',
                'Coleman–Liau index',
                'LIX']])
    fig = px.bar(t[['Moral Foundation: Care',
                    'Moral Foundation: Harm',
                    'Moral Foundation: Fairness',
                    'Moral Foundation: Cheating',
                    'Moral Foundation: Loyalty',
                    'Moral Foundation: Betrayal',
                    'Moral Foundation: Authority',
                    'Moral Foundation: Subversion',
                    'Moral Foundation: Purity',
                    'Moral Foundation: Degradation', ]].T, title="Fraction of words from different moral foundations")

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.markdown(fr"<p>The general moral foundation of the text is {t['General Moral Foundation'].iloc[0]} </p>",
                unsafe_allow_html=True)

if __name__ == "__main__":

    st.set_page_config(layout="wide",
                       page_title="Fact&Bias",
                       page_icon=":newspaper:")

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    st.markdown("<h2 style='text-align: center; color: black;'>Uncover the Truth, Unmask the Biases - Explore 67K websites with Fact&Bias!</h2>",
                unsafe_allow_html=True)

    news_src = st_searchbox(
        search,
        label="Enter and select the news source from here.",
        key = "news_searchbox",
        placeholder="https://www.bbc.com/news",
    )

    bottom_half_landing = st.empty()
    with bottom_half_landing.container():
        st.markdown("<h3 style='color: black;'>Get to know your media source's -</h3><div><h4><ul class='columns' data-columns='2'><li>Factuality</li><li>Propaganda and Identity Framing Techniques used</li><li>Political Bias</li><li>Morality of text used</li><li>Political Stance</li><li>Complexity of text used</li></ul></h4></div>",
                    unsafe_allow_html=True)

    valid_url(news_src)

    if VALID_SRC:
        if news_src[-1] == "/":
            news_src = news_src[:-1]
        print(news_src)

        main_empty = st.empty()
        with main_empty.container():
            bottom_half_landing.empty()
            with st.spinner('Scraping...'):
                results = tp.make_request(news_src)
                print(results)
            # results = tp.aggr_scores(results)

            # biasfig = tp.plotbias(results['bias_results'])
            # factfig = tp.plotfact(results['factuality_results'])

            # identity_results, persuasion_results = tp.get_parq(news_src = news_src)
            # is_identity_persuasion = True if identity_results else False

            # if not is_identity_persuasion:
            #     st.write("Identity Framing and Persuasion Results were not found in the database. Displaying Factuality and Bias Results only.")
            #     plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
            # else:
            #     fig_col1, fig_col2 = st.columns(2)

            #     identfig = tp.plotiden(identity_results)
            #     persfig = tp.plotpers(persuasion_results)

            plot_results(results)

            is_reanalyse = st.button("Reanalyse", key = "reanalysebutton", on_click = valid_url(news_src))
            # plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
            # plot_ident_pers(identfig, persfig)

        if is_reanalyse:
            main_empty.empty()

            with main_empty.container():
                with st.spinner("Reanalyzing..."):
                    results = tp.make_request(news_src, True)
                plot_results(results)

