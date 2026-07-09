import logging
import pandas as pd 
import re
from Config import *

# This preprocessing module is part of the inherited starter prototype.
# It is intentionally not production-ready and may be improved by students.
logger = logging.getLogger(__name__)
# hardcoded file paths 
def get_input_data():
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df = pd.concat([df1, df2])
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    # use TYPE_COLS instead of CLASS_COL because multilabel classification is required
    df["y"] = df[Config.TYPE_COLS]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df

def de_duplication(data):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english": [
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of "
        r"Ireland with its headquarters in Dublin, Ireland\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to "
        r"Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, "
        r"New Zealand and other countries\.?",
    ],
    "german": [
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
        r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht "
        r"mit Sitz in Dublin, Irland\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services "
        r"für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und "
        r"anderen Ländern\.?",
    ],
    "french": [
        r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
        r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le "
        r"siège est à Dublin, en Irlande\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei "
        r"aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, "
        r"en Nouvelle-Zélande et dans d'autres pays\.?",
    ],
    "spanish": [
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
        r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la "
        r"legislación de Irlanda con su sede en Dublín, Irlanda\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei "
        r"a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, "
        r"Nueva Zelanda y otros países\.?",
    ],
    "italian": [
        r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
        r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi "
        r"irlandesi con sede a Dublino, Irlanda\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i "
        r"proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova "
        r"Zelanda e altri paesi\.?",
    ],
    "portuguese": [
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
        r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis "
        r"da Irlanda, com sede em Dublin, Irlanda\.?",
        r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para "
        r"Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova "
        r"Zelândia e outros países\.?",
    ],
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = r"(On.{30,60}wrote:)"
    pattern_3 = r"(Re\s?:|RE\s?:)"
    pattern_4 = r"(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = r"(\s?\*\*\*\*\*\(PHONE\))*$"
    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"
    

    # -------- start processing ticket data

    tickets = data["message_id"].value_counts()

    for t in tickets.index:
        #print(t)
        df = data.loc[data['message_id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df[Config.INTERACTION_CONTENT]:
            if pd.isna(ic):
                ic = ""
            else:
                ic = str(ic)
            # print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            #print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["message_id"] == t, "ic_deduplicated"] = ic_deduplicated
    data[Config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data

def noise_remover(df):
    noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()
    noise_1 = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"dear ((customer)|(user))",
        r"dear",
        r"(hello)|(hallo)|(hi )|(hi there)",
        r"good morning",
        r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        r"thank you for contacting us",
        r"thank you for your availability",
        r"thank you for providing us this information",
        r"thank you for contacting",
        r"thank you for reaching us (back)?",
        r"thank you for patience",
        r"thank you for (your)? reply",
        r"thank you for (your)? response",
        r"thank you for (your)? cooperation",
        r"thank you for providing us with more information",
        r"thank you very kindly",
        r"thank you( very much)?",
        r"i would like to follow up on the case you raised on the date",
        r"i will do my very best to assist youin order to give you the best solution",
        r"could you please clarify your request with following information:in this matter",
        r"we hope you(( are)|('re)) doing ((fine)|(well))",
        r"i would like to follow up on the case you raised on",
        r"we apologize for the inconvenience",
        r"sent from my huawei (cell )?phone",
        r"original message",
        r"customer support team",
        r"(aspiegel )?se is a company incorporated under the laws of ireland with its "
        r"headquarters in dublin, ireland.",
        r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device "
        r"owners in",
        r"canada, australia, new zealand and other countries",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)",
    ]
    for noise in noise_1:
        #print(noise)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(noise, " ", regex=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(r'\s+', ' ', regex=True).str.strip()
    # print("df: ",df)
    
    # good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    # df = df.loc[df.y1.isin(good_y1)]
    #print(df.shape)
    return df

def clean_text_columns(df):
    """Full text-cleaning pipeline shared by both training and inference."""
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def fill_missing_labels(df, target_cols, fill_value=None) :
    # Missing target labels are assigned Not Applicable instead of dropping the rows, since y3/y4 are absent for many ticket
    
    fill_value = fill_value if fill_value is not None else Config.MISSING_LABEL_FILL
    df = df.copy()
    for col in target_cols:
        df[col] = df[col].fillna(fill_value)
    return df


def filter_rare_classes(df, target_cols, min_samples) :
    # drop the labels that are too rare to learn and cannot be used for stratify
    
    df = fill_missing_labels(df, target_cols)
    before = len(df)

    for col in target_cols:
        counts = df[col].value_counts()
        rare = counts[counts < min_samples]
        if len(rare) > 0:
            logger.info("filter_rare_classes: dropping classes in '%s' with < %d samples: %s",col,min_samples,rare.to_dict())
        keep_values = counts[counts >= min_samples].index
        df = df[df[col].isin(keep_values)]

    logger.info("filter_rare_classes: %d -> %d rows", before, len(df))
    return df
