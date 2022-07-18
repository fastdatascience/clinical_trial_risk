import re
import traceback

import pandas as pd
import pdfkit
from airium import Airium
from dash import dcc

from util import graph_callbacks
from util.constants import EXPLANATION_OPTIONS
from util.country_output import pretty_print_countries


def yes_no(x):
    if x:
        return "yes"
    return "no"


def generate_pdf(export_pdf_button_clicks, data, columns, tertiles_data, tertiles_columns, score,
                 tokenised_pages, condition, condition_to_pages,
                 phase, phase_to_pages,
                 sap, sap_to_pages,
                 effect_estimate,
                 effect_estimate_to_pages,
                 num_subjects,
                 num_subjects_to_pages,
                 countries,
                 country_to_pages,
                 simulation,
                 simulation_to_pages,
                 word_cloud,
                 log1, log2,
                 ):
    """
    Generates an export report PDF summary of the risk assessment of the protocol

    :param export_pdf_button_clicks:
    :param data:
    :param columns:
    :param tertiles_data:
    :param tertiles_columns:
    :param score:
    :param tokenised_pages:
    :param condition:
    :param condition_to_pages:
    :param phase:
    :param phase_to_pages:
    :param sap:
    :param sap_to_pages:
    :param effect_estimate:
    :param effect_estimate_to_pages:
    :param num_subjects:
    :param num_subjects_to_pages:
    :param countries:
    :param country_to_pages:
    :param simulation:
    :param simulation_to_pages:
    :param word_cloud:
    :param log1:
    :param log2:
    :return: PDF file in bytes and the file name
    """
    try:
        if export_pdf_button_clicks == 0 or columns is None or data is None:
            return [None]

        word_count = sum([len(t) for t in tokenised_pages])

        df_main = pd.DataFrame()
        df_main["Parameter"] = ["Page count", "Word count", "Average words per page", "Condition", "Phase",
                                "Has the Statistical Analysis Plan been completed?",
                                "Has the Effect Estimate been disclosed?", "Number of subjects",
                                "Countries of investigation", "Trial uses simulation for sample size?"]
        df_main["Value (approved by user if applicable)"] = [f"{len(tokenised_pages)} pages", f"{word_count} words",
                                                             f"{word_count / len(tokenised_pages):.1f} words",
                                                             condition, phase, yes_no(sap),
                                                             yes_no(effect_estimate), num_subjects,
                                                             pretty_print_countries(countries, False),
                                                             yes_no(simulation)]
        df_main["Value found by AI"] = [None, None, None, condition_to_pages["prediction"],
                                        phase_to_pages["prediction"],
                                        yes_no(sap_to_pages["prediction"]),
                                        yes_no(effect_estimate_to_pages["prediction"]),
                                        num_subjects_to_pages["prediction"],
                                        pretty_print_countries(country_to_pages["prediction"], False),
                                        yes_no(simulation_to_pages["prediction"])]
        df_main["Confidence"] = [None, None, None, f"{100 * condition_to_pages['score']:.1f}%", None,
                                 f"{100 * sap_to_pages['score']:.1f}%",
                                 f"{100 * effect_estimate_to_pages['score']:.1f}%",
                                 None, None, f"{100 * simulation_to_pages['score']:.1f}%"]
        main_table_html = df_main.to_html(index=False)
        main_table_html = re.sub(r'>(?:NaN|None)<', '', main_table_html)

        df = pd.DataFrame()
        for col in columns:
            col_name = col['name']
            column_data = [r[col['id']] for r in data]
            df[col_name] = column_data

        df_tertiles = pd.DataFrame()
        for col in tertiles_columns:
            col_name = col['name']
            column_data = [r[col['id']] for r in tertiles_data]
            df_tertiles[col_name] = column_data
        tertiles_table_html = df_tertiles.to_html(index=False)

        risk_category = df["Score"].iloc[-1]
        pathology = df["Value"].iloc[1]
        table_html = df.to_html(index=False)
        table_html = re.sub(r'>(?:NaN|None)<', '', table_html)

        file_name = df["Parameter"].iloc[0]
        pdf_file_name = re.sub(".pdf", "", "analysis_" + file_name) + ".pdf"
        doc_title = f"Analysis of {file_name}"

        text_log = ""
        for text in log1 + log2:
            if type(text) is str:
                text_log += text + "<br/>"

        a = Airium()
        # Generating HTML file
        a('<!DOCTYPE html>')
        with a.html(lang="en"):

            with a.head():
                a.meta(charset="utf-8")
                a.title(_t=doc_title)
                with a.style():
                    a("""body {
  background-color: #ffffff;
  margin: 5%;
  font-family: "PT Sans";
  /* "Noto Serif",serif */
}
h1, h2, h3, h4 {
    color: #188385;
     font-weight:normal; 
}
/* tr:nth-child(even) {background: #CCC}
tr:nth-child(odd) {background: #FFF} */
table{
 border-collapse:collapse;
}
th { text-align: left; background: #c7f9cc; font-weight: normal; }
div {
                    page-break-inside: avoid;
                }""")

            with a.body():
                with a.span(style="font-size:10pt;position:absolute; top:0px; left:0px;"):
                    with a.a(href="https://protocols.fastdatascience.com", target="new"):
                        with a.span(style="color:black;"):
                            a("protocols.fastdatascience.com")
                # with a.span(style="font-size:10pt;position:absolute; top:0px; right:0px;"):
                #     with a.a(href="https://fastdatascience.com", target="new"):
                #         a.img(
                #             src="https://raw.githubusercontent.com/fastdatascience/logos/master/logo_transparent_background.png",
                #             width="150px")
                a.br()
                with a.h1():
                    a(doc_title)
                with a.h2():
                    a(f"This is a {risk_category}-risk {pathology} trial")
                a(f"The risk level is derived from a total score of <b>{score}</b> on a 100-point scale.")
                a.br()
                a("Word cloud of informative terms from the document:")
                a.br()
                a.img(src=word_cloud, width="50%")
                # a(f"<img src=\"{word_cloud}\"/>")
                with a.h2():
                    a(f"Key information about this protocol")
                a(main_table_html)
                with a.h2():
                    a("Risk calculation spreadsheet")
                a("The table below shows how the risk of this protocol was calculated. Protocols are scored according to a simple linear formula between 0 and 100, where 100 would be a perfect low-risk protocol and 0 would be a high-risk protocol. Each parameter extracted in the table on the left is entered into a spreadsheet and contributes to the scoring with an associated weight. For example, by far the strongest indicator that a protocol is low-risk is the presence of a statistical analysis plan.")
                a(table_html)

                with a.h2():
                    a("Sample size tertiles")
                a("The model characterises trials as small, medium and large according to the number of participants. Since early phase trials are smaller than later trials, tertiles are used to define what counts as e.g. a small HIV Phase I trial. The table of tertiles is given below.")
                a(tertiles_table_html)
                with a.h2():
                    a("How the protocol was analysed")
                a("This is a log of the analysis of the text which was carried out by the protocol analysis tool, with an explanation of the scoring.")
                a(text_log)
            for title, what_to_display in EXPLANATION_OPTIONS:
                graph_fig, graph_surtitle, contexts = graph_callbacks.display_breakdown_graph_by_pages_in_document(
                    what_to_display, tokenised_pages, condition_to_pages,
                    country_to_pages,
                    None,
                    effect_estimate_to_pages,
                    None,
                    None,
                    num_subjects_to_pages,
                    phase_to_pages,
                    sap_to_pages,
                    simulation_to_pages)
                graph_html = graph_fig.to_html(config={'displayModeBar': False})

                def translate_dash_html_to_airium(contexts):
                    for ct in contexts:
                        if type(ct) is str:
                            a(re.sub(" - click the legend to hide", "", ct))
                        else:
                            if ct.children:
                                a(f"<{type(ct).__name__}>{str(ct.children)}</{type(ct).__name__}>")
                            else:
                                a(f"<{type(ct).__name__} />")

                if title == title.lower():
                    title = title.title()
                with a.div():
                    with a.h2():
                        a("Explanation: " + title)
                    if graph_surtitle:
                        translate_dash_html_to_airium(graph_surtitle)
                    a(graph_html)
                    if contexts:
                        translate_dash_html_to_airium(contexts)

        doc_html = str(a)
        # with open("/tmp/tmp.html", "w", encoding="utf-8") as f:
        #     f.write(doc_html)

        pdf_bytes = pdfkit.from_string(doc_html)

    except:
        print("Error generating PDF\n", traceback.format_exc())

    return [dcc.send_bytes(pdf_bytes, pdf_file_name)]
