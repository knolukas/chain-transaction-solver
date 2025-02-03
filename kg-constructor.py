import getpass
import os
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(refresh_schema=False)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

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
wko_1 = """
Der französische Unternehmer FR1 bestellt beim französischen Unternehmer FR2 Ware. 
Dieser hat die Ware nicht lagernd und bestellt sie beim österreichischen Unternehmer AT. 
AT beauftragt einen Spediteur die Ware direkt zu FR1 nach Frankreich zu befördern. 
FR2 gibt seine französische UID-Nummer bekannt.
"""
documents = [Document(page_content=sachverhalt)]

# """
# First approach without filter
# """
# llm_transformer = LLMGraphTransformer(llm=llm)
# graph_documents = llm_transformer.convert_to_graph_documents(documents)
# print(f"Nodes:{graph_documents[0].nodes}")
# print(f"Relationships:{graph_documents[0].relationships}")

"""
Second approach with three-tuple
"""
allowed_relationships = [
    ("Person", "SPOUSE", "Person"),
    ("Person", "NATIONALITY", "Country"),
    ("Person", "WORKED_AT", "Organization"),
]

llm_transformer_props = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Firm"],
    allowed_relationships=["ORDERS_FROM", "DELIVERS_TO"],
    node_properties=["country"],
    relationship_properties=["good"]
)

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

def delete_graph():
    query = "MATCH (n) DETACH DELETE n"
    graph.query(query)
    print("Graph deleted successfully")


delete_graph()
