import asyncio
import json
import os

import pandas as pd
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import rag_wrapper
import itertools

# Initialisiere einen globalen Counter für die Graph-IDs
graph_id_counter = itertools.count(start=20)


def get_next_graph_id():
    return f"graph_{next(graph_id_counter)}"


load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(refresh_schema=False)

# # Alle PDFs aus dem "files/" Ordner laden
# pdf_files = [f for f in os.listdir("files") if f.endswith(".pdf")]
#
# documents = []
# for pdf in pdf_files:
#     pdf_loader = PyPDFLoader(os.path.join("files", pdf))
#     documents.extend(pdf_loader.load())
#
# # Teile den Text in kleinere Abschnitte (Chunking)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)
#
# # Vektor-DB laden
# vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
# vectorstore.save_local("files")
#
# # FAISS-Datenbank laden (Sicherheitsoption beachten)
# vectorstore = FAISS.load_local("files/", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#
# # Retriever aus der Vektor-Datenbank erstellen
# retriever = vectorstore.as_retriever()

# LLM initialisieren
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# # Prompt für den RAG-Workflow
# prompt_template = """Nutze die folgenden Dokumente, um die Frage zu beantworten:
# {context}
#
# Frage: {question}
# Antwort:"""
# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#
# # RAG-Chain definieren
# rag_chain = LLMChain(llm=llm, prompt=prompt)
#
#
# # Wrapper-Funktion zum Abrufen relevanter Dokumente
# def rag_with_context(question: str):
#     docs = retriever.get_relevant_documents(question)
#     context = "\n".join([doc.page_content for doc in docs])
#     return rag_chain.invoke({"context": context, "question": question})


text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

sachverhalt = """
Unter Berücksichtigung der Angaben in den gegenständlichen vier Anmeldungen, der
Ermittlungen des Zollamtes Linz Wels (nachfolgend: Zollamt), der im gerichtlichen und
verwaltungsbehördlichen Abgabenverfahren hervorgekommenen Unterlagen, insb. des
vom Zollamt dem Bundesfinanzgericht (nachfolgend: BFG) übersendeten rechtskräftig
gewordenen Urteils des Landgerichts Bochum (nachfolgend: LG Bochum) sowie aufgrund
der Angaben und der Verantwortung der Parteien des verwaltungsbehördlichen und
gerichtlichen Abgabenverfahrens wird vom nachfolgend angeführten Verfahrensgang
und Sachverhalt ausgegangen:
Die Bf ist ein Speditionsunternehmen und beantragte mit vier Anmeldungen als indirekte
Vertreterin der in den Anmeldungen angegebenen Empfängerin B (UID-Nr. BEbb) unter
Verwendung der der Bf erteilten Sonder-UID-Nr. die Überführung von Tonerkartuschen mit
Verfahrenscode 42 (nachfolgend: VC 42) in den zoll- und steuerrechtlich freien Verkehr
unter Befreiung von der Einfuhrumsatzsteuer (nachfolgend: EUSt) mit unmittelbar daran
anschließender innergemeinschaftlicher (nachfolgend: ig) Lieferung.
Betroffen sind die Anmeldungen mit der CRN 11ATdd vom 05.04.2011, mit der
CRN 11ATee vom 27.04.2011, mit der CRN 11ATff vom 19.05.2011 und mit der CRN
11ATgg vom 15.06.2011.
Die angemeldeten Waren wurden in Österreich antragsgemäß vom Zollamt in den zoll-
und steuerrechtlich freien Verkehr übergeführt. Dabei wurde EUSt-Freiheit gewährt.
In den Anmeldungen war als Versender/Ausführer jeweils die C (nachfolgend: D) ein in
Istanbul ansässiges Unternehmen, als Empfänger jeweils die im Mitgliedstaat Belgien
ansässige B angegeben.
Als Bestimmungsland war jeweils Deutschland angegeben. Die Ware sollte jeweils an die
E, Adresse3 (nachfolgend: F - UID-Nr. DEhh) geliefert werden. Als Lieferbedingung wurde
in den Frachtbriefen jeweils "FCA-Istanbul", in den Anmeldungen "FCA" angegeben.
Das Zollamt  hat der Bf mit Schreiben vom 06.12.2013 mitgeteilt, dass es u.a. diese
vier Überführungen in den zoll- und steuerrechtlich freien Verkehr mit Befreiung von der
EUSt mit unmittelbar daran anschließenden ig Lieferungen mit VC 42 prüfen wolle. Das
Zollamt hat die Bf darum ersucht, sie möge deshalb zum Nachweis der Steuerbefreiung
der Einfuhren diverse im Schreiben namentlich angeführte Nachweise vorlegen.
Die Verordnung (EU) 904/2010 hat die Zusammenarbeit der Verwaltungsbehörden und
die Betrugsbekämpfung auf dem Gebiet der Mehrwertsteuer zum Inhalt. Das Zollamt hat
sich danach im Rahmen eines Informationsaustausches an das Finanzamt Bochum
gewendet und diesem im Wesentlichen mitgeteilt, die Bf habe Tonerkartuschen mit sehr
hohen Werten verzollt. Die Waren seien im Rahmen eines Reihengeschäftes an die F
gegangen. Rechnungen der B an die F, irgendwelche sonstigen Aufträge, Bestellungen
udgl. gebe es nicht. Die genannten Unternehmen seien im Konkurs. Aus den vorliegenden
Unterlagen ergebe sich, dass möglicherweise nicht die F, sondern ein Empfänger in
"Adresse4" der letzte Empfänger gewesen sei. Es gebe keine Nachweise darüber, in
welchem Zusammenhang der zweite Empfänger F und ein allfälliger dritter Empfänger
stehen. Die Bf habe die B als Käufer und die F als Empfänger angegeben bzw. dass die
Waren im Auftrag der B zur F in Deutschland und nicht nach Belgien transportiert worden
seien.
Die Bf verfüge je Verzollung über einen Frachtbrief mit Übernahmebestätigung
durch die F und durch die B und über einen selbst erstellten Lieferschein mit einen
Übernahmevermerk von der F, aber über keinerlei Bestätigung der Übernahme eines
eventuell weiteren Empfängers an der "Adresse4". 
Die Bf habe bisher nicht mit stichhaltigen Unterlagen nachweisen können, dass die F der
letzte Empfänger gewesen sei. Die Bf habe als letzten Empfänger bei zwei Verzollungen
"Adresse5" und bei zwei Verzollungen "Adresse6" angegeben, obwohl es in Adresse7 eine
F gar nicht gebe. Möglicherweise befinde sich in Adresse7 ein weiterer Empfänger. Das
Zollamt habe daher keine sicheren Nachweise darüber, wo die Lieferungen tatsächlich
hingegangen sind. 
Das Finanzamt Bochum hat geantwortet und das Zollamt im Wesentlichen auf Ergebnisse
des Ermittlungsverfahrens der "Ermittlungskommission G" gegen eine europaweit
agierende Täterbande aus dem Bereich des Umsatzsteuerbetruges verwiesen. Danach
sei die F ein Missing-Trader, die B ein In-Out-Buffer gewesen. Die Täterbande habe sie als
Funktionsträger eingebunden, um einen Nettowarenbezug zu ermöglichen und um die
Waren innerhalb einer rein fiktiven Rechnungskette (Missing-Trader - zwei Buffer) zu
verbilligen. Die Waren seien letztlich über einen Distributor vermarktet worden. Alle nach
außen hin gerichteten Handlungen der Funktionsträger seien Scheinhandlungen gewesen.
Insb. sei deren Rechnungslegung und Buchhaltung zentral gesteuert worden. Tatsächlich
habe es auf der Ebene der Funktionsträger keine wirtschaftlichen Betätigungen gegeben.
Die Waren seien nie nach Belgien oder zur F geliefert worden. Die Täterbande um
die Haupttäter H, J und K sei inzwischen durch das LG Bochum rechtskräftig zu
Freiheitsstrafen verurteilt worden. Die UID-Nr. der F sei in Deutschland rückwirkend mit
01.04.2011 begrenzt worden.
Aus dem Ermittlungsbericht und dem Urteil des LG Bochum ergibt sich, dass die L
(nachfolgend: M) als Buffer-II und die N (nachfolgend: O) als Distributor ihren Sitz an der
Adresse4, hatten.
Geschäftsführer der M, deren Gegenstand der Handel mit elektronischen Geräten, Hard-
und Software, Druckerverbrauchsmaterialien war, war K 
J war Geschäftsführer der O und der B. Die B handelte mit Computerersatzteilen,
Peripherie- und Verbrauchermaterial. Gegenstand der O war der
Groß- und Einzelhandel mit Waren aller Art, insb. mit EDV-Zubehör,
Verbrauchsmaterialien, Elektro- bzw. Elektronikgeräten. Das Betrugsmodell
diente den Tatbeteiligten dazu, Umsatzsteuer (nachfolgend: USt) zu verkürzen, indem
das Umsatzsteuerrecht systemwidrig zur Verbilligung von Waren ausgenutzt wurde. Das
Betrugsmodell diente den Tatbeteiligten auch dazu, eine zusätzliche Einnahmequelle in
Form von Bargeldbeträgen zur persönlichen Verwendung zu erlangen.
Auf den Gegenstand bezogen initiierten die Tatbeteiligten die in Belgien ansässige
Scheinfirma B als In-Out-Buffer. Die Zwischenschaltung dieser Gesellschaft ermöglichte
den umsatzsteuerrechtlich unbelasteten Nettoeinkauf. Die B fakturierte sodann eine
innergemeinschaftliche - und damit umsatzsteuerfreie - Lieferung an die F, ebenfalls eine
Scheinfirma. Der Missing-Trader F hatte im Betrugssystem einzig und allein die Aufgabe,
die Nettoeingangsrechnung der B in eine gleich hohe Bruttoausgangsrechnung an die
M, eine weitere Scheinfirma als Buffer im Betrugssystem umzuschreiben. Die M konnte
dadurch die in der von der F umgeschriebenen Rechnung ausgewiesene Mehrwertsteuer
als Vorsteuer lukrieren, wobei die F als Missing-Trader die in der umgeschriebenen
Rechnung ausgewiesene Mehrwertsteuer nicht an das Finanzamt abgeführt hat. Auf
der Ebene des Distributors O wirkte sich der Taterfolg in der Form eines verbilligten
Warenabverkaufs aus.
Nach Ausweis der Verwaltungsakten verfügt die Bf über die von ihr an die B gestellte
Speditionsabrechnungen für getätigte Verzollungsleistungen mit entsprechenden
Zahlungsbelegen, jedoch nicht - abgesehen vom E-Mail-Verkehr - über kaufmännische
Unterlagen darüber, ob bzw. in welcher Geschäftsbeziehung die B zur F, die B zur M
bzw. die F zur M standen. Ferner verfügt die Bf über Abfragen (FinanzOnline) der UID-
Nr. der Abnehmerin F, die zum Zeitpunkt der Abfertigungen vorgenommen und zu diesem
Zeitpunkt mit gültig bestätigt worden waren.
Die vier gegenständlichen Einfuhren wurden zwischen dem 05.04.2011 und
dem 15.06.2011 vorgenommen. Die UID-Nummer der F wurde rückwirkend beschränkt.
Das Zollamt hat bei der Bf betreffend das Jahr 2011 eine Prüfung hinsichtlich der
Überführung von Waren in den zoll- und steuerrechtlich freien Verkehr mit Befreiung
von der EUSt bei unmittelbar danach anschießender ig Lieferung durchgeführt. Aus der
darüber aufgenommenen Niederschrift ergibt sich, dass von der Betriebsprüfung die vier
gegenständlichen Anmeldungen der B mit Erklärung der UID-Nr. der F als Empfängerin
der Waren jeweils in Feld 44 der Anmeldungen sowie die im Zuge der Prüfung von der Bf
eingeforderten bzw. von ihr vorgelegten Unterlagen geprüft wurden.
A us der Sicht des Betriebsprüfungsberichtes ist die Bf z ur Führung des Nachweises des
Vorliegens der Voraussetzungen für die Befreiung der Einfuhren von der EUSt bei jeweils
unmittelbar daran anschließender ig Lieferung wie folgt vorgegangen:
- Sie hat jeweils die Verzollungspapiere, den CMR-Frachtbrief, die Rechnung,
den vollständigen Steuerbescheid etc. an die B in Belgien mit einem als
„Warenübernahmebestätigung“ oder „INFO-Datenblatt“ bezeichneten Begleitschreiben
übersendet, die B um Originalbestätigung des CMR-Frachtbriefs (in Feld 24 „Gut
empfangen“) und um dessen Retournierung per Post an die Niederlassung Adresse8 der
Bf in Deutschland ersucht;
- Sie hat ein selbst erstelltes, als „Lieferschein“ bezeichnetes Schriftstück und eine
weitere Ausfertigung des CMR-Frachtbriefes an die F übersendet, die F um Bestätigung
im Original beider Belege für die Übernahme der Waren und deren Rücksendung an
die Niederlassung Adresse8 der Bf in Deutschland ersucht.
Festzuhalten ist nochmals, dass nach Ausweis der Akten Ermittlungen des Zollamtes
tatsächlich darauf hindeuten, dass nicht die B oder die F die endgültigen Empfänger
der Tonerkartuschen waren, sondern dass die Waren für einen Abnehmer in
Adresse7 bestimmt waren.
Mit Schriftsatz vom 04.03.2014 hat die Bf zum Betriebsprüfungsbericht Stellung
genommen und den Antrag gestellt, sie weder abgaben- noch finanzstrafrechtlich zur
Verantwortung zu ziehen. Sie billige ebenso wie die österreichische Zollbehörde keinen
Steuerbetrug. Sie dürfe für mögliche Steuerbetrügereien - nur weil die österreichischen
Zollbehörden das Unionsrecht falsch auslegen würden - aber nicht in die Haftung
genommen werden.
Mit Bescheid vom 19.03.2014 teilte das Zollamt der Bf im Wesentlichen mit, für sie sei
bei der Überführung von eingangsabgabepflichtigen Waren in den zollrechtlich freien
Verkehr mit anschließender ig Lieferung und der Annahme der Anmeldungen gemäß
Art. 201 Abs. 1 Buchstabe a) und Abs. 3 ZK iVm § 2 Abs. 1 ZollR-DG EUSt im Betrage
von insgesamt € 157.429,31 entstanden. Die angeführten Eingangsabgaben seien gemäß
Art. 220 ZK nachträglich buchmäßig erfasst worden und würden ihr gemäß Art. 221 ZK
mitgeteilt.   
Nach Wiedergabe des Sachverhaltes und der Vorbringen hat das Zollamt zur Sache des
Verfahrens und zu den Vorbringen erwogen. Es hat seine Entscheidung i m Wesentlichen
damit begründet, bei der Betriebsprüfung seien die vier gegenständlichen Anmeldungen
über ig Erwerbe durch die B mit der F als Empfängerin der Waren überprüft worden. Die
Bf sei in den Anmeldungen mit ihrer Sonder-UID-Nr. für Spediteure als Anmelder und
Vertreter im indirekten Vertretungsverhältnis aufgetreten und habe demnach das Vorliegen
der Voraussetzungen für das jeweils angemeldete Zollverfahren nachzuweisen gehabt
aber nicht nachweisen können. Das Zollamt schließt seine Begründung mit Ausführungen
zur deswegen entstandenen EUSt-Schuld, zur Bf als Schuldnerin der EUSt und zur
ermessensgerechten Inanspruchnahme von Gesamtschuldnern .
Gegen den Bescheid hat die Bf in offener Frist mit Schriftsatz vom 27.03.2014
Beschwerde erhoben und u.a. den sachrelevanten Antrag gestellt, es möge der Bescheid
vom 19.03.2014 wegen Rechtswidrigkeit seines Inhalts aufgehoben werden.
D as Zollamt hat über die Beschwerde mit seiner Beschwerdevorentscheidung
(nachfolgend: BVE) vom 14.05.2014 entschieden und die Beschwerde als unbegründet
abgewiesen. Es hat nach einer gerafften Wiedergabe des Sachverhaltes und der
Vorbringen zur Sache des Verfahrens erwogen und zum Beschwerdevorbringen Stellung
genommen.
Gegen die BVE hat die Bf in offener Frist mit Schriftsatz vom 23.05.2014 den Antrag auf
Entscheidung über die Beschwerde durch das BFG gestellt und beantragt, das BFG möge
- den Bescheid des Zollamtes vom 19.03.2014 und die BVE des Zollamtes vom
14.05.2014 aufheben;
- eine Entscheidung über die Beschwerde durch alle Mitglieder des Berufungssenates
treffen;
- eine mündliche Verhandlung anberaumen und
- die Kosten des Verfahrens dem Beschwerdegegner auferlegen.
Unter dem Eindruck des Urteils des EuGH vom 02.06.2016 in der Rs C-226/2014
(Eurogate Distribution - nachfolgend: Eurogate-II) hat die Bf ihren Vorlageantrag mit
Schriftsatz vom 01.11.2016 ergänzt. Sie hat am 02.11.2016 u.a. den Vorlagebeschluss des
Hessisches FG vom 29.09.2015 und die diesbezügliche Stellungnahme der Europäischen
Kommission  nachgereicht um zu bemerken, es sei die Vorlage von Fragen an den EuGH
nach den CILFIT-Kriterien - auch wenn das BFG seine Auffassung für richtig halte -
dennoch geboten, wenn andere ernstzunehmende Rechtsauffassungen zu Rechtsfragen
existieren. Der EuGH sei mit vielen Rechtsfragen im Zusammenhang mit der ig Lieferung,
aber bislang nicht mit einer einzigen zur Steuerfreiheit der ig Anschlusslieferung befasst
worden. Es wundere dabei doch sehr, gerade angesichts des Art. 6 Abs. 3 UStG 1994, der
auf das Steuersystem der ig Lieferung und gerade nicht auf das Zollrecht verweise, das
hier dennoch angewandt werde.
Am 11.11.2016 hat die Bf beim BFG vorgesprochen. Dazu hat sie ein Protokoll verfasst
und es als Anlage zum Schriftsatz vom 25.11.2016 vorgelegt.
Am 09.01.2017 hat das Zollamt dem BFG das nach der  VO 904/2010/EU eingeholte
rechtskräftige Urteil des LG Bochum vom 20.02.2013 übersendet. Das Urteil ist in der
Strafsache u.a. gegen H, J und K bzw. zu den im Gegenstand beteiligten Firmen B, F, M
und O ergangen. Im Wesentlichen haben danach H, J und K ein Umsatzsteuerkarussell
mit einem Firmenkonstrukt aufgebaut und laufend ausgebaut.
Die Bf wurde vom BFG mit Schreiben vom 10.01.2017 eingeladen, das Urteil beim
Zollamt einzusehen.
Die Bf hat sich dazu mit Schreiben vom 30.01. und vom 28.02.2017 geäußert.
Am 20.03.2017 wurde eine mündliche Verhandlung abgehalten. Der Vertreter der
Beschwerdeführerin und der Vertreter des Zollamtes konnten kontradiktorisch zum
Sachverhalt Stellung nehmen und Standpunkte bzw. Rechtsmeinungen austauschen.
Im Wesentlichen stimmen die Parteien darin überein, dass die Bf im konkreten
Beschwerdefall nicht Lieferer im Rahmen der sich an die Verzollung anschließenden
ig Lieferung war. Die Verfahrensparteien stimmen auch darin überein, dass der
Belegnachweis, sofern man den Betrug ausblendet, in Ordnung gewesen wäre. Die
Bf bringt vor, dass die ursprünglich an den EuGH zu stellenden Fragen, durch die
neu vorgeschlagenen Fragen (Schriftsatz vom 25.11.2016) überholt sind. Die Bf
nimmt den Antrag auf Kostenerstattung bzw. Auferlegung der Kosten an das Zollamt
zurück. Sollte die Beschwerde im Hinblick auf die EUSt abgewiesen werden, kann die
Entscheidung über die Abgabenerhöhung, zumal über diese Rechtsfrage Revisionen
beim VwGH behängen, zur Vermeidung weiterer Revisionen zurückgestellt und zu einem
späteren Zeitpunkt mit einem eigenen Erkenntnis erledigt werden. Die Parteien verweisen
auf ihre schriftlichen Eingaben. Das Zollamt beantragt die Abweisung der Beschwerde. Die
Bf beantragt die Stattgabe der Beschwerde bzw. andernfalls die Vorlage der im Vorbringen
vom 25.11.2016 angesprochenen drei Fragen an den EuGH.
"""
sachverhalt_2 = """
Mit Bescheiden vom 10.05.2013 wurden der Beschwerdeführerin (BF) Säumniszuschläge
von je 2 % betreffend Umsatzsteuer 2010 (idHv. 7.658,17 Euro) und betreffend
Umsatzsteuer 2011 (idHv. 2.865,43 Euro) vorgeschrieben.
Begründend wurde ausgeführt, dass die Festsetzungen erforderlich gewesen seien,
weil die angeführten Abgabenschuldigkeiten nicht innerhalb der Frist (15.02.2011 bzw.
15.02.2012) entrichtet worden seien. 
Dagegen richtete sich die Berufung/Beschwerde der BF vom 05.06.2013 und wurde als
Begründung seitens der steuerlichen Vertretung ausgeführt:
"Im Rahmen der bei unserer Mandantin durchgeführten Umsatzsteuerprüfung über
die Jahre 2010 und 2011 hat sich herausgestellt, dass die BF (Steuer Nr. 1111)
einige Rechnungen an die SSS Trading GmbH (Steuer Nr. 33333) als steuerfreie
Ausfuhrlieferung qualifiziert und somit die Rechnungen über den gelieferten KK ohne
österreichische Umsatzsteuer (10%) ausgestellt hat. Die Betriebsprüferin, Frau ABP, hat
die von unserer Mandantin in diesem Zusammenhang vertretene Rechtsansicht, dass
diese Lieferungen gemäß § 6 UStG steuerfrei sind, nicht geteilt.
Bei diesen Umsätzen, welche von der BF als steuerfreie Ausfuhrlieferungen qualifiziert
wurden, handelte es sich jeweils um ein sogenanntes Reihengeschäft. Dies deshalb, als
mehrere Unternehmer über denselben Gegenstand Umsatzgeschäfte abgeschlossen
haben und der KK im Rahmen der Beförderung oder Versendung unmittelbar vom ersten
Unternehmer an den letzten Abnehmer gelangt ist.
Bei Reihengeschäften kommt es für die Bestimmung des umsatzsteuerlichen Lieferortes
darauf an, wem die sog. 'bewegte' Lieferung zugerechnet wird, denn nur diese kann
eine steuerfreie Ausfuhrlieferung sein. Unter Hinweis auf die EuGH Urteile vom
6.4.2006, EMAG, C-245/04 und vom 16.12.2010 Euro Tyre, C-430/09 ist die BF davon
ausgegangen, dass in diesen Fällen ihr die bewegte Lieferung ins Drittland zugerechnet
werden kann. Nach Ansicht der Betriebsprüfung ist jedoch die bewegte Lieferung der
SSS Trading GmbH zuzurechnen und daher liegt zwischen der BF und der SSS Trading
GmbH eine ruhende Lieferung vor, welche als Inlandslieferung in Österreich steuerbar und
steuerpflichtig ist.
Aufgrund der Rechtsprechung des EuGH ist die BF daher in diesen Fällen - nach Ansicht
der Betriebsprüfung fälschlicherweise - davon ausgegangen, dass der BF die bewegte
Lieferung zuzurechnen war und deshalb wurden die Lieferungen steuerfrei behandelt.
Diesen Umstand haben wir der Prüferin im Rahmen der Prüfung sowohl mündlich
erläutert als ihr auch schriftlich mitgeteilt. Leider hat sie die Prüfung ohne formelle
Schlussbesprechung abgeschlossen und somit unsere Rechtsansicht auch nicht in einer
Niederschrift über die Schlussbesprechung aufgenommen.
Als die Prüferin im Rahmen der Prüfung ihre Bedenken gegen die von unserer Mandantin
vertretene Rechtsansicht angemerkt hat, wurde mit ihr vereinbart, dass die Berichtigung
dieser Rechnungen mittels Sammelberichtigung für 2010 und 2011 erfolgt und dass die
Umsatzsteuer durch Übertragung des Vorsteuerguthabens von der SSS Trading GmbH
auf das Abgabenkonto der BF erfolgen wird.
Aus diesem Grund wurden die Rechnungen berichtigt und die Umsatzsteuererklärungen
2010 bzw. 2011 der BF sowie jene der SSS Trading GmbH korrigiert.
Das daraus resultierende Vorsteuerguthaben der SSS Trading GmbH wurde auf
das Abgabenkonto der BF überrechnet, um die aus dieser Korrektur entstandene
Umsatzsteuerzahllast zu begleichen.
Dies wurde von unserer Mandantin unmittelbar nach einer Besprechung mit der Prüferin
durchgeführt. Dies wird Ihnen die Umsatzsteuerprüferin auch bestätigen können, denn sie
hat diesen Umstand bei der Veranlagung auch berücksichtigt.
Der Republik Österreich ist daher zu keinem Zeitpunkt ein Schaden bzw. Nachteil
entstanden, denn, wäre der Sachverhalt ursprünglich bereits so von unserer Mandantin
qualifiziert worden, wäre ebenfalls das Vorsteuerguthaben mittels Überrechnung auf das
Abgabenkonto der BF übertragen worden.
Kein grobes Verschulden (§ 217 Abs. 7 BAO)
Das Antragsrecht auf Herabsetzung bzw. Nichtfestsetzung von Säumniszuschlägen
setzt voraus, dass den Abgabepflichtigen kein grobes Verschulden an der Säumnis
trifft. Nimmt ein zur Selbstberechnung Verpflichteter die Selbstberechnung vor und
entrichtet er (zeitgerecht) den selbst berechneten Betrag, so ist für § 217 Abs. 7 BAO
ausschlaggebend, ob ihn an einer Fehlberechnung, also z.B. wie im vorliegenden Fall eine
zu niedrige Berechnung, ein grobes Verschulden trifft.
Gemäß Rz 975 der Richtlinien zur Abgabeneinhebung ist dies beispielsweise dann nicht
der Fall, wenn der Selbstberechnung eine vertretbare Rechtsansicht zugrunde liegt. War
die Rechtsansicht unvertretbar, so ist dies für die Anwendung des § 217 Abs. 7 nur bei
Vorsatz oder grober Fahrlässigkeit schädlich (RAE, Rz 975; UFS 11.7.2007, RV /0664-
L/05).
Ein (grobes) Verschulden wird daher idR etwa dann zu verneinen sein, wenn der
Abgabepflichtige der Selbstberechnung eine Rechtsprechung des EuGH zugrundelegt.
Da unsere Mandantin im vorliegenden Fall die Rechtsansicht des EuGH vertreten hat,
liegt uE keinerlei grobes Verschulden vor und daher stellen wir unter Bezugnahme auf §
217 Abs. 7 BAO den Antrag, den ersten Säumniszuschlag für die Umsatzsteuer 2010 und
Umsatzsteuer 2011 mangels groben Verschuldens nicht festzusetzen und die jeweiligen
Bescheide aufzuheben."
Das Finanzamt wies die Berufung/Beschwerde als unbegründet ab mit der Begründung,
dass angesichts der in den Umsatzsteuererklärungen ausgewiesenen hohen Zahllasten
nicht von einem bloß minderen Grad des Versehens nach § 217 Abs. 7 BAO ausgegangen
werden könne.
Dagegen wendete sich die BF in ihrem Vorlageantrag vom 24.07.2013 und führte
ergänzend aus: Die Höhe der Zahllast könne nicht dafür ausschlaggebend sein,
ob ein minderer Grad des Versehens nach § 217 Abs. 7 BAO vorliege, wenn eine
vertretbare Rechtsansicht vorliege.
Selbst wenn der falschen Berechnung eine unvertretbare Rechtsauffassung zugrunde
liegen würde, wäre § 217 Abs. 7 BAO anwendbar, allerdings nur dann, wenn leichte
Fahrlässigkeit vorliege (vgl. RAE Rz 975 letzter Satz). 
In sachverhaltsmäßiger Hinsicht wird festgestellt, dass es bei der BF aufgrund einer
(aus Sicht der Außenprüfung) falschen rechtlichen Beurteilung eines Teiles ihrer
Lieferungen als steuerfreie Ausfuhrlieferungen bei Beteiligung an einem Reihengeschäft
verspätet zu einer Erhöhung der Inlandsumsätze für 2010 und 2011, und damit zu einer
Nachforderung von Umsatzsteuer kam. Die BF vermeinte, ihre Rechtsmeinung aus der
EuGH-Rechtsprechung ableiten zu können, die Prüferin teilte diese Rechtsmeinung nicht
und wurden die steuerpflichtigen Umsätze erhöht.
Die betroffenen Rechnungen wurden seitens der BF berichtigt und die
ausgewiesene Umsatzsteuer von der Rechnungs- und Leistungsempfängerin (als
Vorsteuerabzugsberechtigte) unmittelbar danach auf das Konto der BF überrechnet
(Buchung vom 27.12.2012 idHv 526.179,91 Euro). 
Für die verspätet entrichtete Umsatzsteuer 2010 und 2011 wurden nach § 217 BAO
(automatisiert) Säumniszuschläge festgesetzt.
Der Antrag auf mündliche Verhandlung vor dem Senat wurde im Verfahren vor dem BFG
zurückgenommen.
"""
beispiel_7a = """
Sachverhalt (aus der Sicht des österreichischen Unternehmers U2):
Dem österreichischen Unternehmer U2 wird die Ware vom Schweizer Unternehmer U1 verrechnet. Der österreichische 
Unternehmer verrechnet die Ware weiter an den deutschen Unternehmer U3. Die Ware gelangt aber direkt vom Schweizer 
Unternehmer U1 an den italienischen Empfänger U4. Transport wird durch U2 veranlasst.
"""
reihegeschaeft_bsp = """Ein österreichischer Unternehmer U4 (=Empfänger) bestellt bei seinem österreichischen Lieferanten 
U3 (=2. Erwerber) eine Maschine. Dieser wiederum bestellt die Maschine beim österreichischen 
Großhändler U2 (=1. Erwerber). Da der Großhändler U2 die Maschine nicht auf Lager hat, bestellt 
er diese beim österreichischen Produzenten U1 (=Erstlieferant).
Der österreichische Großhändler U2 holt die Maschine vom österreichischen 
Produzenten U1 ab und liefert diese direkt an den österreichischen Unternehmer U4."""

wko_1 = """
Der französische Unternehmer FR1 bestellt beim französischen Unternehmer FR2 Ware. 
Dieser hat die Ware nicht lagernd und bestellt sie beim österreichischen Unternehmer AT. 
AT beauftragt einen Spediteur die Ware direkt zu FR1 nach Frankreich zu befördern. 
FR2 gibt seine französische UID-Nummer bekannt.
"""


# class RAGLLMWrapper:
#     """Wrapper, um RAG als LLM für LLMGraphTransformer nutzbar zu machen."""
#
#     def __init__(self, rag_chain):
#         self.rag_chain = rag_chain  # Die RAG-Kette speichern
#
#     def invoke(self, prompt):
#         """Diese Methode wird aufgerufen, wenn LLMGraphTransformer eine Antwort erwartet."""
#         return self.rag_chain.run(prompt)  # RAG-Kette für den Prompt ausführen

def delete_graph():
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")


def format_graph_documents(graph_docs):
    formatted_data = []

    for graph_doc in graph_docs:  # Iteriere über die Liste
        formatted_data.append({
            "nodes": [
                {"id": node.id, "type": node.type, "properties": node.properties}
                for node in graph_doc.nodes
            ],
            "relationships": [
                {
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type": rel.type,
                    "properties": rel.properties,
                }
                for rel in graph_doc.relationships
            ],
            "source_document": {
                "metadata": graph_doc.source.metadata,
                "content": graph_doc.source.page_content,
            },
        })

    return json.dumps(formatted_data, indent=4, ensure_ascii=False)


# async def process_graph():
#     data = await llm_transformer_props.aconvert_to_graph_documents(documents)
#     graph.add_graph_documents(data)
#     print(data)


def main():
    print("Hello, World!")


# allowed_nodes_good = ["Good", "Person", "Company", "Entrepreneur", "Carrier"]
# allowed_nodes_with_graph_id_good = [f"{node}:{graph_id}" for node in allowed_nodes_good]
# allowed_relationships_good = [
#     (f"{start}:{graph_id}", rel, f"{end}:{graph_id}") for start, rel, end in [
#         ("Good", "ORDERED_FROM", "Company"),
#         ("Good", "ORDERED_FROM", "Entrepreneur"),
#         ("Good", "ORDERED_FROM", "Person"),
#         ("Good", "DELIVERED_TO", "Company"),
#         ("Good", "DELIVERED_TO", "Entrepreneur"),
#         ("Good", "DELIVERED_TO", "Person"),
#         ("Good", "DELIVERED_FROM", "Person"),
#         ("Good", "DELIVERED_FROM", "Person"),
#         ("Good", "DELIVERED_FROM", "Person"),
#         ("Company", "INSTRUCTS", "Carrier"),
#         ("Person", "INSTRUCTS", "Carrier"),
#         ("Entrepreneur", "INSTRUCTS", "Carrier"),
#         ("Good", "COLLECTED_FROM", "Company"),
#         ("Good", "COLLECTED_FROM", "Entrepreneur"),
#         ("Good", "COLLECTED_FROM", "Person"),
#     ]
# ]
# node_properties_good = ["country", "UID", "name"]
# relationship_properties_good = ["date", "serial_id"]

# rag_llm = RAGLLMWrapper(rag_chain)


# llm_transformer_props_good = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=allowed_nodes_with_graph_id_good,
#     allowed_relationships=allowed_relationships_good,
#     node_properties=node_properties_good,
#     relationship_properties=relationship_properties_good,
# )

additional_instructions = ("system: Add the name of the transported good as "
                           "relationship property \"transported_good\".")

project_directory = "H:/Users/Lukas/OneDrive/Masterarbeit - LLMs in VAT - Knogler Lukas/"
df = pd.read_excel(project_directory + "Beispiele_Reihengeschäfte.xlsx")


def process_text_with_graph_transformer(text):
    """Wendet den LLM-Graph-Transformer an und fügt die Graph-ID hinzu."""
    graph_id = get_next_graph_id()  # Nächste Graph-ID holen
    allowed_nodes = ["Person", "Company", "Entrepreneur", "Carrier"]
    allowed_nodes_with_graph_id = [f"{node}:{graph_id}" for node in allowed_nodes]
    allowed_relationships = [
        (f"{start}:{graph_id}", rel, f"{end}:{graph_id}") for start, rel, end in [
            ("Company", "ORDERS_FROM", "Company"),
            ("Company", "ORDERS_FROM", "Entrepreneur"),
            ("Company", "ORDERS_FROM", "Person"),
            ("Entrepreneur", "ORDERS_FROM", "Company"),
            ("Entrepreneur", "ORDERS_FROM", "Entrepreneur"),
            ("Entrepreneur", "ORDERS_FROM", "Person"),
            ("Person", "ORDERS_FROM", "Company"),
            ("Person", "ORDERS_FROM", "Entrepreneur"),
            ("Person", "ORDERS_FROM", "Person"),
            ("Company", "DELIVERS_TO", "Company"),
            ("Company", "DELIVERS_TO", "Entrepreneur"),
            ("Company", "DELIVERS_TO", "Person"),
            ("Entrepreneur", "DELIVERS_TO", "Company"),
            ("Entrepreneur", "DELIVERS_TO", "Entrepreneur"),
            ("Entrepreneur", "DELIVERS_TO", "Person"),
            ("Carrier", "DELIVERS_TO", "Company"),
            ("Carrier", "DELIVERS_TO", "Entrepreneur"),
            ("Carrier", "DELIVERS_TO", "Person"),
            ("Person", "DELIVERS_TO", "Company"),
            ("Person", "DELIVERS_TO", "Entrepreneur"),
            ("Person", "DELIVERS_TO", "Person"),
            ("Company", "INSTRUCTS", "Carrier"),
            ("Person", "INSTRUCTS", "Carrier"),
            ("Entrepreneur", "INSTRUCTS", "Carrier"),
            ("Company", "COLLECTS_FROM", "Company"),
            ("Company", "COLLECTS_FROM", "Entrepreneur"),
            ("Company", "COLLECTS_FROM", "Person"),
            ("Entrepreneur", "COLLECTS_FROM", "Company"),
            ("Entrepreneur", "COLLECTS_FROM", "Entrepreneur"),
            ("Entrepreneur", "COLLECTS_FROM", "Person"),
            ("Carrier", "COLLECTS_FROM", "Company"),
            ("Carrier", "COLLECTS_FROM", "Entrepreneur"),
            ("Carrier", "COLLECTS_FROM", "Person"),
            ("Person", "COLLECTS_FROM", "Company"),
            ("Person", "COLLECTS_FROM", "Entrepreneur"),
            ("Person", "COLLECTS_FROM", "Person"),
        ]
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes_with_graph_id,
        allowed_relationships=allowed_relationships,
        prompt=get_prompt()
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs

json_string = """{{
  "parsed": {{
    "nodes": [
      {{
        "id": "FR1",
        "type": "Company"
      }},
      {{
        "id": "FR2",
        "type": "Company"
      }},
      {{
        "id": "AT",
        "type": "Company"
      }},
      {{
        "id": "Good",
        "type": "Product"
      }},
      {{
        "id": "Austria",
        "type": "Country"
      }},
      {{
        "id": "France",
        "type": "Country"
      }},
      {{
        "id": "Delivery_1",
        "type": "Delivery"
      }}
    ],
    "relationships": [
      {{
        "source_node_id": "F1",
        "source_node_type": "Company",
        "target_node_id": "FR2",
        "target_node_type": "Company",
        "type": "ORDERS"
      }},
      {{
        "source_node_id": "FR2",
        "source_node_type": "Company",
        "target_node_id": "AT",
        "target_node_type": "Company",
        "type": "ORDERS"
      }},
      {{
        "source_node_id": "Delivery_1",
        "source_node_type": "Delivery",
        "target_node_id": "AT",
        "target_node_type": "Company",
        "type": "INITIATED_BY"
      }},
      {{
        "source_node_id": "Delivery_1",
        "source_node_type": "Delivery",
        "target_node_id": "Carrier_A",
        "target_node_type": "Carrier",
        "type": "TRANSPORTED_BY"
      }},
      {{
        "source_node_id": "Good",
        "source_node_type": "Product",
        "target_node_id": "Delivery_1",
        "target_node_type": "Delivery",
        "type": "PART_OF"
      }},
      {{
        "source_node_id": "FR1",
        "source_node_type": "Company",
        "target_node_id": "France",
        "target_node_type": "COUNTRY",
        "type": "LOCATED_IN"
      }},
      {{
        "source_node_id": "FR2",
        "source_node_type": "Company",
        "target_node_id": "France",
        "target_node_type": "COUNTRY",
        "type": "LOCATED_IN"
      }},
      {{
        "source_node_id": "AT",
        "source_node_type": "Company",
        "target_node_id": "Austria",
        "target_node_type": "COUNTRY",
        "type": "LOCATED_IN"
      }}
    ]
  }},
  "parsing_error": null
}}"""

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to reconstruct a chain transaction in a simple but also comprehensible way, making it\n"
    "accessible for a further processing to identify the correct tax obligations.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'HAS_ORDERED', use more general and timeless relationship types "
    "like 'ORDER'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "Company DE", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "DE", "the company DE"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "DE" as the entity ID.\n'
    "- **Maintain Relationship Consistency**: When extracting relationships, it's vital to "
    "ensure consistency.\n"
    'If a relation, such as "DELIVER", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "sends", "delivers", "carries", "transports"),'
    "always use the allowed identifier from the allowed relationships for that relationship throughout the "
    'knowledge graph. In this example, use "DELIVER" as the name of the relationship.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Completeness\n"
    "Each chain transaction consists of at least these nodes and relations:\n"
    '- At least 3 or more different companies and/or entrepreneurs\n'
    '- One product/good that is the object of the transaction\n'
    '- At least one node (company, entrepreneur) that is responsible for the transport (e.g organizes the transport '
    'or instructs another party in doing so\n'
    '- One product/good that is the object of the whole transaction\n'
    '- Make sure that every node has a relationship according to the respective allowed relationship per node type.\n'
    "## 5. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination.\n"
    "## 6. Explanatory example\n"
    "Der französische Unternehmer FR1 bestellt beim französischen Unternehmer FR2 Ware."
    "Dieser hat die Ware nicht lagernd und bestellt sie beim österreichischen Unternehmer AT."
    "AT beauftragt einen Spediteur die Ware direkt zu FR1 nach Frankreich zu befördern."
    "FR2 gibt seine französische UID-Nummer bekannt.\n"
    "Solution:\n" + json_string
)


def get_prompt(
        additional_instructions: str = "",
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                additional_instructions
                + " Tip: Make sure to answer in the correct format and do "
                  "not include any explanations. "
                  "Use the given format to extract information from the "
                  "following input: {input}",
            ),
        ]
    )


missing_good_prompt = ("system: If the good is not explicitly stated, simply generate a node called \"good\" "
                       "and incorporate it into the graph. Same counts for the delivery, simply generate a node called"
                       "delivery.")


def process_text_with_graph_transformer_v2(text):
    """Wendet den LLM-Graph-Transformer an und fügt die Graph-ID hinzu."""
    graph_id = get_next_graph_id()  # Nächste Graph-ID holen
    allowed_nodes = ["Company", "Delivery", "Invoice", "Order", "Product", "Country", "Carrier"]
    allowed_relationships = [
        ("Company", "ORDERS", "Company"),
        ("Delivery", "INITIATED_BY", "Company"),
        ("Delivery", "TRANSPORTED_BY", "Company"),
        ("Delivery", "TRANSPORTED_BY", "Carrier"),
        ("Company", "LOCATED_IN", "Country"),
        ("Product", "PART_OF", "Delivery"),
    ]
    node_properties = ["country", "UID", "name"]
    relationship_properties = ["date", "transported_good"]

    llm_transformer_props = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        prompt=get_prompt()
    )

    docs = llm_transformer_props.convert_to_graph_documents([Document(page_content=text)])

    return docs

output = process_text_with_graph_transformer_v2(df.loc[0, 'text'])
# Wende die Funktion auf die DataFrame-Spalte an
df.loc[0:2, 'graph'] = df.loc[0:2, 'text'].apply(process_text_with_graph_transformer)
df.loc[0:5, 'graph2'] = df.loc[0:5, 'text'].apply(process_text_with_graph_transformer_v2)  # standard prompt
df.loc[0:0, 'graph1'] = df.loc[0:0, 'text'].apply(lambda x: process_text_with_graph_transformer_v2(x))  # hier mit neuer prompt
# Füge die generierten Graphen zu `graph` hinzu
delete_graph()
df.loc[0:0, 'graph3'].apply(lambda x: print(x))
df.loc[0:0, 'graph1'].apply(lambda x: graph.add_graph_documents(x))

# additional_instructions_good = ("system: First, identify the good of interest and then interpret the whole transaction"
#                                 "as a pov from the good.")
# documents = [Document(page_content=reihegeschaeft_bsp + additional_instructions)]
# data = llm_transformer_props.convert_to_graph_documents(documents)


# Run the async function in the event loop
# asyncio.run(process_graph())
# # Print GraphDocuments
# print(format_graph_documents(data))

if __name__ == "__main__":
    main()

# llm_transformer_tuple = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=["Person", "Country", "Organization"],
#     allowed_relationships=allowed_relationships,
# )
graph_documents_filtered = llm_transformer_props.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")

graph.add_graph_documents(graph_documents_filtered)

# """
# Third approach with additional node properties
# """
# llm_transformer_props = LLMGraphTransformer(
#     llm=llm,
#     allowed_nodes=["Person", "Country", "Organization"],
#     allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
#     node_properties=["born_year"],
# )
# graph_documents_props = llm_transformer_props.convert_to_graph_documents(documents)
# print(f"Nodes:{graph_documents_props[0].nodes}")
# print(f"Relationships:{graph_documents_props[0].relationships}")
#
# graph.add_graph_documents(graph_documents_props)
