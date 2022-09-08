##############################
# import of relevant modules #
##############################
import pandas as pd


#####################################################
# Class definition for the NER and ABSA predictions #
#####################################################
class APC_NER_Prediction:

    def __init__(self):

        # lists to keep track of sentence manipulations
        self.raw_sentence = []
        self.preprocessed_sentences = []

        # NER module variables
        self.tagger = None
        self.splitter = None

        # data frame for ner related operations and results
        self.ner_df = None

        # data frame for apc (aspect polarity classification) related operations and results
        self.df_apc = None

        # data frame to combine and operate on ner and apc related aspects
        self.ner_apc_df = None

        # APC related variable
        self.sent_classifier = None

        # list to keep track of the sentiment predictions
        self.apc_predictions = []

        # variable for the user input text - initialized with some default values
        self.input_text = "Deutsche Bank AG (German pronunciation: [ˈdɔʏtʃə ˈbaŋk ʔaːˈɡeː] (listen)) is\
         a German multinational investment bank and financial services company headquartered in Frankfurt,\
          Germany, and dual-listed on the Frankfurt Stock Exchange and the New York Stock Exchange. "

        # variable to track which entity should be automatically detected - default ORG
        self.entity = 'ORG'


    def apc(self):
        """
        function to automatically infer the aspect based sentiment
        on already preprocessed text data
        :input: preprocessed text in format <preceeding text>[ASP]<entitiy>[ASP]<succeeding text>
        :output: sentences with predicted sentiment on provided entities
        """

        for sample in self.preprocessed_sentences:
            # call actual PyABSA APC model for inference
            result = self.sent_classifier.infer(sample, print_result=False)
            # below line serves to not show any Errors on sentences that do not comprise an entity that should be
            # used for APC inference --> model would return error of not provided with a [ASP]entity[ASP]
            result['text'] = result['text'].replace(" ERROR, Please WRAP the target aspects!", '')
            self.apc_predictions.append(result)

        return self.apc_predictions

    def ner_detection(self):
        """
        function to automatically detect entities on provided input text
        1. transform text into sentences
        2. detect dedicated entities in sentences (ORG)
        3. transform detected entities into required format for ABSA
        :input: plain text that is not preprocessed
        :output: preprocessed text with detected entities that are transformed into required ABSA format
        """

        # use splitter to split text into list of sentences (input text is provided on document level)
        self.raw_sentence = self.splitter.split(self.input_text)

        # predict tags for sentences
        self.tagger.predict(self.raw_sentence)

        # iterate through all segmented sentences and enrich the found and relevant entities with [ASP] tags
        df = pd.DataFrame()
        for sentence in self.raw_sentence:
            sent = sentence.text
            span_list = []
            ent_list = []
            ent_name = []
            # handle sentences that comprise a relevant entity and sentences that do not comprise any relevant entity
            if len(sentence.get_spans('ner')) > 0:
                for i in sentence.get_spans('ner'):
                    # transform detected entities within sentences according to the user selection on the desired Entity
                    if i.tag in self.entity:
                        print(i.tag)
                        # span_list.append([i.start_position, i.end_position])
                        ent_list.append(i.tag)
                        ent_name.append(i.text)
                        ent_tokens = [token.idx-1 for token in i]
                        span_list.append(ent_tokens)
                        sent_split = sent.split()
                        for ix, element in enumerate(ent_tokens):
                            if len(ent_tokens) == 1:
                                sent_split[element] = f"[ASP]{i.text.split()[ix]}[ASP]"
                            elif ix == 0:
                                sent_split[element] = f"[ASP]{i.text.split()[ix]}"
                            elif ix == len(ent_tokens)-1:
                                sent_split[element] = f"{i.text.split()[ix]}[ASP]"
                        sent = " ".join(sent_split)
            else:
                span_list.append('nothing found')
                ent_list.append('nothing found')
                ent_name.append('nothing found')
            tmp = pd.DataFrame()
            tmp['span'] = [span_list]
            tmp['ents'] = [ent_list]
            tmp['name'] = [ent_name]
            tmp['sent'] = [sent]
            self.preprocessed_sentences.append(sent)
            df = pd.concat([df, tmp], axis=0)

        # track all detected NER tags, spans, ... in one global df for latter manipulation
        df = df.reset_index(drop=True)
        self.ner_df = df

    def entity_formatter(x):
        """
        function to manipulate each sentence that was processed by NER and APC into a format that can later be
        displayed to the user in the UI
        1. transform detected aspect (via NER and APC) into format: (ASPECT, SENTIMENT, COLORCODE)
        2. transform sentence into format that can be used by the html_formatter
           --> tuples for aspects vizualization
           --> stings for plain text displaying
        :input: df row with sentences and all respective detected NER + APC attributes
        :output: formatted sentence for latter html/markdown embedding into the app
        """

        # 1. aspect encoding into UI display format (latter html formatting)
        tmp = []
        if len(x['name']) > 0:
            for i in x.aspect:
                tmp_tpl_idx = x.aspect.index(i)
                if x.sentiment[tmp_tpl_idx] == 'Positive':
                    tmp_tpl = (x.aspect[tmp_tpl_idx], x.ents[tmp_tpl_idx], "#008000")
                elif x.sentiment[tmp_tpl_idx] == 'Negative':
                    tmp_tpl = (x.aspect[tmp_tpl_idx], x.ents[tmp_tpl_idx], "#DC143C")
                elif x.sentiment[tmp_tpl_idx] == 'Neutral':
                    tmp_tpl = (x.aspect[tmp_tpl_idx], x.ents[tmp_tpl_idx], "#696969")
                tmp.append(tmp_tpl)

            # 2. get entities and replace them with a placeholder
            sent_split = x.sent.split('[ASP]')
            count = 0
            for ix, i in enumerate(sent_split):
                if i in x.aspect:
                    sent_split[ix] = tmp[count]
                    count += 1
        else:
            sent_split = x.sent

        return sent_split

    def assembler(self):
        """
        function to orchestrate the whole pipeline
        1. apply NER on input text (sentence splitting, NER, APC formatting, ...)
        2. apply APC on preprocessed sentences
        3. concat NER and APC results for each sentence in one df
        4. format each sentence within the df into succeeding html/markdown embedding in the app
        :input: raw text
        :output: df with final format of sentences for latter html/markdown embedding
        """
        print('----------------------------------------')
        print('NER & APC pipeline orchestration started')
        print('----------------------------------------')

        # 1.
        APC_NER_Prediction.ner_detection(self)
        # 2.
        apc_predictions = APC_NER_Prediction.apc(self)

        # 3.
        # format the detected entities for the UI
        self.df_apc = pd.DataFrame.from_dict(apc_predictions)
        self.ner_apc_df = pd.concat([self.df_apc, self.ner_df], axis=1)

        # 4.
        # self.ner_apc_df['drop'] = self.ner_apc_df.ents.apply(lambda x: 'drop' if x == ['nothing found'] else 'keep')
        # self.ner_apc_df = self.ner_apc_df[self.ner_apc_df['drop'] == 'keep']
        print(self.ner_apc_df)
        print(self.ner_apc_df.keys())


        self.ner_apc_df['formatted_text'] = self.ner_apc_df.apply(lambda x: APC_NER_Prediction.entity_formatter(x),
                                                                  axis=1)

        print('----------------------------------------')
        print('NER & APC pipeline orchestration finished')
        print('----------------------------------------')

