############
# Citation #
############
"""
  @misc{YangL2022,
  title = {PyABSA: Open Framework for Aspect-based Sentiment Analysis},
  author = {Yang, Heng and Li, Ke},
  doi = {10.48550/ARXIV.2208.01368},
  url = {https://arxiv.org/abs/2208.01368},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information
  sciences},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}}
"""

##############################
# import of relevant modules #
##############################
import streamlit as st
import ABSA_HTML_Formatter

from pyabsa import APCCheckpointManager
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from ABSA_App_HelperFunctions import APC_NER_Prediction



#####################
# APC model loading #
#####################
@st.cache(allow_output_mutation=True)
def load_apc_model(model_name):
    sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint=model_name,
                                                                    auto_device=True)
    return sent_classifier

#####################
# NER model loading #
#####################
@st.cache(allow_output_mutation=True)
def load_ner_model(model_name):

    tagger = SequenceTagger.load(model_name)
    splitter = SegtokSentenceSplitter()

    return tagger, splitter


##############################
# APP appearance definitions #
##############################

# app title definition
st.title("Aspect Based Sentiment Analysis")
# explaination of app intention and general guidance
st.write(
    "This is a demonstration of the capabilities of our aspect-based sentiment model.\
     In order to use this demonstrator you would first need to select the entity for which you would like to\
     infer the sentiment and thereafter provide some input text which you would like to automatically\
     process. Once the process ran through you can see the results at the bottom ob the website.")

# definition of selectable choices for the user
st.subheader("Select an entity you would like to get the sentiment for")

# logic implementation for latter NER (determine which entities will be selectable)
# define entity selection layout
left, middle, right = st.columns(3)
with left:
    entOrg = st.checkbox('Organization', True)
with middle:
    entPer = st.checkbox('Person', True)
with right:
    entLoc = st.checkbox('Location', True)

entity = []
if entPer:
    entity.append('PER')
if entOrg:
    entity.append('ORG')
if entLoc:
    entity.append('LOC')
print(entity)

# definition of a custom text input for the user
st.subheader("Text input for aspect-based sentiment analysis")
input_text = st.text_area("Here you can simply input the text you would like to process by the model.")
print('----------------------------------------')
print(f"INPUT TEXT: {len(input_text)}")
print('----------------------------------------')

# Start of the acutal input processing part - once provided by the user
with st.spinner("processing your provided input"):
    # load NLP models for sentiment and ner predictions - change to "flair/ner-english-large"
    sent_classifier = load_apc_model('english')
    tagger, splitter = load_ner_model('ner')

    # create the object that is responsible for text processing
    predictor = APC_NER_Prediction()

    # check if user content has already been provided
    if len(input_text) == 0:
        # initialize the relevant NER functions for text processing
        predictor.tagger = tagger
        predictor.splitter = splitter
        # initialize the APC classifier (sentiment)
        predictor.sent_classifier = sent_classifier
        # initialize the entity that should be referenced for sentiment predictions (default is ORG)
        predictor.entity = entity
    else:
        # initialize the relevant NER functions for text processing
        predictor.tagger = tagger
        predictor.splitter = splitter
        # initialize the APC classifier (sentiment)
        # using this backbone model:
        # --> https://huggingface.co/microsoft/deberta-v3-base (https://github.com/microsoft/DeBERTa)
        # --> paper: https://arxiv.org/abs/2111.09543
        predictor.sent_classifier = sent_classifier
        # initialize the text variable that provides the basis for all processing steps
        predictor.input_text = input_text#.replace('\n', '')
        # initialize the entity that should be referenced for sentiment predictions (default is ORG)
        predictor.entity = entity

        # initialize the assembler object that orchestrates in the following way
        # text input --> apply NER --> apply ABSA --> result formatting
        predictor.assembler()

        # transform the ABSA result into consumable html format for the UI
        text_to_display = predictor.ner_apc_df.formatted_text.tolist()
        text_to_display = [e for i in text_to_display for e in i]


        # display the transformed ABSA result in the UI
        st.markdown(ABSA_HTML_Formatter.get_annotated_html(text_to_display),
                    unsafe_allow_html=True)




#############
# DEBUGGING #
#############

# sent_classifier = load_apc_model('english')
# tagger, splitter = load_ner_model('ner')
# predictor = APC_NER_Prediction()
# predictor.input_text = "Deutsche Bank AG (German pronunciation: [ˈdɔʏtʃə ˈbaŋk ʔaːˈɡeː] (listen)) is a German multinational investment bank and financial services company headquartered in Frankfurt, Germany, and dual-listed on the Frankfurt Stock Exchange and the New York Stock Exchange. "
# predictor.input_text = "General - YASH TRADING & FINANCE LTD. - Reporting Of Initial Disclosure To Be Made By Entities Considered As ''Large Corporate'' Under The SEBI Circular SEBI/HO/DDHS/CIR/P/2018/144 Dated 26-11-2018"
# predictor.tagger = tagger
# predictor.splitter = splitter
# predictor.sent_classifier = sent_classifier
# predictor.entity = ['ORG']
# predictor.assembler()
# text_to_display = predictor.ner_apc_df.formatted_text.tolist()
# text_to_display = [e for i in text_to_display for e in i]
# st.markdown(ABSA_HTML_Formatter.get_annotated_html(text_to_display),
#             unsafe_allow_html=True)

################################
# Exemplary text for ABSA demo #
################################

# Deutsche Bank is about to launch their best product ever, while HSBC is launching their worst product. This is a sentence without an entity and it is just for showcasing. I love Berlin because Capgemini has an office there and it is a great company.
# Unilever is a great company in the CPRD sector and just recently launched a genius new product together with ALDI.
# The 150-year-old Deutsche Bank AG (DB) announced on Sunday a much-awaited radical transformation that the embattled company hopes will make it leaner and meaner and able to survive in the long term.

