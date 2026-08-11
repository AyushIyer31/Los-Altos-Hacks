"""Regenerate the PET Lab Media Outreach Kit PDF with updated founder/collaboration
details applied consistently to all 50 pitches + front/back matter.

Changes vs. the original kit:
  - Two co-founders (Ayush Iyer & Abhinav Iyer) everywhere, Abhinav email in signatures
  - Team described as "more than 14 undergraduate and high school researchers"
  - Collaboration line: UC Santa Cruz + Santa Clara University (HPC / Nautilus),
    faculty mentorship from NYU
  - Interview offer now "an interview with the founders" (both are co-founders)
  - Tehran municipal/mayoral office intentionally NOT included
Contacts, subjects, sources, per-pitch hooks are preserved verbatim from the original.
"""
import os
from fpdf import FPDF
import matplotlib as _mpl

FD = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "PET_Lab_Media_Outreach_Kit_50_Contacts.pdf")

TEAL = (26, 94, 102)
INK = (24, 24, 24)
GREY = (95, 95, 95)
LIGHTBG = (232, 240, 240)
ROWALT = (244, 247, 247)
WHITE = (255, 255, 255)
RULE = (208, 214, 214)

M = 16.0
FOOTER = ("Public professional contacts checked July 21, 2026   |   "
          "Recheck before sending   |   Page ")


class PDF(FPDF):
    def header(self):
        self.set_y(8); self.set_x(M)
        self.set_font("S", "B", 7.5); self.set_text_color(*TEAL)
        self.cell(0, 4, "PET LAB   |   MEDIA OUTREACH KIT")
        self.set_y(M)

    def footer(self):
        self.set_y(-13); self.set_font("S", "", 7.2); self.set_text_color(*GREY)
        self.cell(0, 4, FOOTER + str(self.page_no()), align="R")


pdf = PDF(format="letter")
pdf.set_margins(M, M, M)
pdf.set_auto_page_break(True, 15)
pdf.add_font("S", "", os.path.join(FD, "DejaVuSans.ttf"))
pdf.add_font("S", "B", os.path.join(FD, "DejaVuSans-Bold.ttf"))
pdf.add_font("S", "I", os.path.join(FD, "DejaVuSans-Oblique.ttf"))
PW = pdf.w
NX = dict(new_x="LMARGIN", new_y="NEXT")


def H1(t):
    pdf.ln(1.5); pdf.set_x(M); pdf.set_font("S", "B", 17); pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 8.5, t, **NX); pdf.ln(1.5)


def H2(t):
    pdf.ln(1.8); pdf.set_x(M); pdf.set_font("S", "B", 11); pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 5.6, t, **NX); pdf.ln(0.6)


def P(t, size=9.4):
    pdf.set_x(M); pdf.set_font("S", "", size); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.7, t, **NX); pdf.ln(1.4)


def leadpara(lead, rest):
    pdf.set_x(M); pdf.set_font("S", "B", 9.4); pdf.set_text_color(*INK)
    pdf.write(4.7, lead + " ")
    pdf.set_font("S", "", 9.4); pdf.write(4.7, rest); pdf.ln(6.6)


def bullet(text, x=M, width=None):
    if width is None:
        width = PW - 2 * M
    pdf.set_x(x); pdf.set_font("S", "", 9.3); pdf.set_text_color(*INK)
    pdf.cell(4, 4.7, "•")
    old = pdf.l_margin; pdf.set_left_margin(x + 5); pdf.set_x(x + 5)
    pdf.multi_cell(width - 5, 4.7, text, **NX)
    pdf.set_left_margin(old)


def kv(label, value, value_color=INK):
    pdf.set_x(M); pdf.set_font("S", "B", 8.5); pdf.set_text_color(*INK)
    pdf.write(4.4, label + "   ")
    pdf.set_font("S", "", 9.0); pdf.set_text_color(*value_color)
    pdf.write(4.4, value); pdf.ln(5.0)


def count_lines(text, width, size, style=""):
    pdf.set_font("S", style, size)
    words = text.split()
    if not words:
        return 1
    line = ""; n = 1
    for w in words:
        t = (line + " " + w).strip()
        if pdf.get_string_width(t) <= width - 1:
            line = t
        else:
            n += 1; line = w
    return n


# ---------------------------------------------------------------- shared copy
COLLAB = ("We collaborate with UC Santa Cruz and Santa Clara University on "
          "high-performance computing for this work, including the Nautilus "
          "research cluster, and receive faculty mentorship from NYU.")
TEAM_LOCAL = ("a student-led Bay Area research team of more than 14 undergraduate "
              "and high school researchers")
TEAM_NAT = ("a student-led research team of more than 14 undergraduate and high "
            "school researchers")

T = {
    "A": ("We are Ayush Iyer and Abhinav Iyer, co-founders of PET Lab "
          "(Predictive Enzyme Technology Laboratory), " + TEAM_LOCAL +
          " working at the intersection of artificial intelligence, computational "
          "biology, protein engineering, and plastic pollution. " + COLLAB +
          " We are developing computational methods to identify enzyme mutations "
          "that may perform more reliably when temperature, pH, salts, and "
          "wastewater conditions change."),
    "N": ("We are Ayush Iyer and Abhinav Iyer, co-founders of PET Lab "
          "(Predictive Enzyme Technology Laboratory), " + TEAM_NAT +
          " applying artificial intelligence and computational biology to "
          "plastic-degrading enzyme design. " + COLLAB +
          " Our goal is to rank mutations that may improve performance under "
          "changing real-world conditions, while being clear that model predictions "
          "require independent laboratory validation."),
    "S": ("We are Ayush Iyer and Abhinav Iyer, co-founders of PET Lab "
          "(Predictive Enzyme Technology Laboratory). Our student-led team of more "
          "than 14 undergraduate and high school researchers is building "
          "computational models that rank candidate mutations in plastic-degrading "
          "enzymes using protein, mutation, and environmental-condition features. " +
          COLLAB + " The aim is to help prioritize what should be tested "
          "experimentally, not to replace laboratory validation or claim a finished "
          "solution."),
    "C": ("We are Ayush Iyer and Abhinav Iyer, co-founders of PET Lab "
          "(Predictive Enzyme Technology Laboratory), " + TEAM_NAT +
          " studying a practical obstacle in plastic biotechnology: enzymes that "
          "appear promising under controlled conditions may struggle when "
          "temperature, pH, salts, contaminants, and wastewater conditions vary. " +
          COLLAB + " We are using computational models to prioritize mutations for "
          "future testing."),
}
QEND = "Would this be worth a brief conversation or a referral to the right reporter?"
OFFER = ("We can share a one-page fact sheet, short demo, clear visuals, our current "
         "methods and limitations, and an interview with the founders.")
ROUTING = ("We would appreciate guidance to the most appropriate editor or producer "
           "if this address is not the right destination.")
EUREKA_TAIL = ("We can provide our project summary, mentor information, and the "
               "institutional partner who would be responsible for any eligible release.")
OPINION_TAIL = ("The essay would draw on our direct experience building PET Lab and "
                "would disclose our role and current stage of research.")


def make_body(tmpl, closer):
    if closer == "route":
        tail = ROUTING + " " + QEND
    elif closer == "eureka":
        tail = EUREKA_TAIL + " " + QEND
    elif closer == "opinion":
        tail = OPINION_TAIL + " " + QEND
    else:
        tail = OFFER + " " + QEND
    return T[tmpl] + " " + tail


SIG = ["Best,", "Ayush Iyer & Abhinav Iyer", "Co-Founders, PET Lab",
       "[PET Lab website]",
       "Ayush: [Your email]   |   Abhinav: abhinaviyer555@gmail.com   |   [Your phone]"]

# ---------------------------------------------------------------- 50 pitches
# (n, outlet, role, category, route, email, subject, why, greeting, hook, tmpl, closer, source)
PITCHES = [
 (1,"CBS News Bay Area (KPIX)","Assignment Editors","Local broadcast","Direct newsroom","kpixnewsassign.editors@cbs.com",
  "Bay Area student lab uses AI to improve plastic-degrading enzymes",
  "Highest-priority Bay Area TV target; strong fit for a visual local youth-science segment.",
  "Hi CBS News Bay Area Assignment Desk,",
  "This is a Bay Area youth-science story with strong television visuals: student founders, computational protein models, and a practical environmental problem.",
  "A","def","https://www.cbsnews.com/sanfrancisco/kpix5/"),
 (2,"KTVU FOX 2","News Tips Desk","Local broadcast","Direct newsroom","newstips@fox.com",
  "Local story idea: Bay Area students build an AI enzyme research lab",
  "Major Bay Area television newsroom with a public news-tip email.",
  "Hi KTVU News Desk,",
  "PET Lab could make a strong local innovation segment because the story combines Bay Area students, AI, biology, and plastic pollution.",
  "A","def","https://www.ktvu.com/about-us"),
 (3,"NBC Bay Area","Candice Nguyen, Investigative Reporter","Local broadcast","Public journalist email - ask for routing","candice.nguyen@nbcuni.com",
  "Bay Area student researchers tackling plastic pollution with AI",
  "A publicly listed NBC Bay Area journalist who may route a credible local science pitch.",
  "Hi Candice,",
  "We are reaching out because this is a local, evidence-focused story about how young researchers are applying computational tools to an environmental challenge.",
  "A","def","https://www.nbcbayarea.com/investigations/antioch-unified-board-president-supervisor-bullying-complaints/3514713/"),
 (4,"ABC7 News Bay Area","Amanda del Castillo, Reporter","Local broadcast","Public journalist email - ask for routing","Amanda.L.delcastillo@abc.com",
  "ABC7 story idea: Student-led Bay Area lab combines AI and biotechnology",
  "Local television reporter with a public professional email; useful for routing to the right producer.",
  "Hi Amanda,",
  "The project has a clear Bay Area community angle and can be explained visually through our modeling workflow, protein structures, and a short founder interview.",
  "A","def","https://abc7news.com/about/newsteam/amanda-del-castillo"),
 (5,"KQED","News Assignment Desk","Local public media","Direct newsroom","assignmentdesk@kqed.org",
  "KQED pitch: Bay Area students use AI to study plastic-degrading enzymes",
  "Excellent fit for public-interest science, education, climate, and Bay Area innovation coverage.",
  "Hi KQED Assignment Desk,",
  "PET Lab sits at the intersection of public-interest science, environmental education, and student-led Bay Area research.",
  "A","def","https://kqed-helpcenter.kqed.org/s/article/How-can-I-reach-the-newsroom-or-radio-reporters"),
 (6,"San Francisco Chronicle","Metro Desk","Local newspaper","Direct newsroom","metrodesk@sfchronicle.com",
  "Bay Area science pitch: Student founders apply AI to enzyme engineering",
  "Top regional newspaper; local science and technology story with a broader environmental angle.",
  "Hi San Francisco Chronicle Metro Desk,",
  "This is a local science and technology story about students building a research effort rather than simply entering a one-time competition.",
  "A","def","https://www.sfchronicle.com/newsroom_contacts/"),
 (7,"The San Francisco Standard","Susie Cagle, Climate and Investigations Editor","Local digital news","Public editor email","scagle@sfstandard.com",
  "Climate-tech story idea: Bay Area student lab studies plastic-degrading enzymes",
  "Especially relevant to climate, technology, and Bay Area accountability-oriented reporting.",
  "Hi Susie,",
  "We thought this might fit your climate and investigations lens because the project focuses on the gap between promising enzymes and the difficult conditions found in actual waste and water systems.",
  "A","def","https://sfstandard.com/author/susie-cagle/"),
 (8,"San Jose Spotlight","Newsroom","Local digital news","Public newsroom contact","info@sanjosespotlight.com",
  "South Bay youth innovation pitch: AI, enzymes, and plastic waste",
  "South Bay civic newsroom that may value student innovation and local environmental impact.",
  "Hi San Jose Spotlight Newsroom,",
  "PET Lab offers a local youth-innovation story with community relevance, a concrete research process, and an environmental mission.",
  "A","def","https://sanjosespotlight.com/news-tips/"),
 (9,"Pleasanton Weekly","Editor","Local newspaper","Public editor email","editor@pleasantonweekly.com",
  "East Bay student researchers launch AI-powered enzyme project",
  "Highly relevant East Bay community outlet; a realistic first feature that can build credibility.",
  "Hi Pleasanton Weekly Editor,",
  "As an East Bay community story, PET Lab shows how local students are creating a sustained interdisciplinary research project around a global environmental problem.",
  "A","def","https://www.pleasantonweekly.com/news/2017/01/29/pleasanton-weekly-names-new-editor/"),
 (10,"CalMatters","General Newsroom","California news","Public general contact - ask for routing","info@calmatters.org",
  "California student-research story: AI-guided enzymes for plastic waste",
  "Statewide reach; pitch the education, research access, and environmental policy implications.",
  "Hi CalMatters Newsroom,",
  "The California angle is not only the science: it is also how students are accessing advanced computational research and trying to connect it to real waste-management challenges.",
  "A","def","https://calmatters.org/about/contact-us/"),
 (11,"CBS News Sacramento","Newsroom","Regional broadcast","Direct newsroom","news@kovr.com",
  "Northern California student lab uses AI to study plastic-degrading enzymes",
  "Regional CBS affiliate that can amplify a Northern California student-science story.",
  "Hi CBS Sacramento Newsroom,",
  "This Northern California story combines student initiative, biotechnology, AI, and a problem that affects communities far beyond the Bay Area.",
  "A","def","https://www.cbsnews.com/sacramento/contact-us/"),
 (12,"The Sacramento Bee","News Tips","California newspaper","Direct tips inbox","tips@sacbee.com",
  "California innovation pitch: Students use AI for enzyme engineering",
  "Statewide influence and strong education, environment, and California innovation readership.",
  "Hi Sacramento Bee News Desk,",
  "PET Lab could fit your education or environment coverage as a California student-led research initiative with a clear technical and public-interest angle.",
  "A","def","https://www.sacbee.com/news/coronavirus/article247266274.html"),
 (13,"Los Angeles Times","News Tips","National newspaper","Direct tips inbox","tips@latimes.com",
  "Story idea: California student founders bring AI to plastic-degrading enzymes",
  "Major California and national outlet; strongest angle is youth science plus environmental technology.",
  "Hi Los Angeles Times Newsroom,",
  "The story connects California student innovation with the growing use of AI in biology and the difficult question of whether plastic-degrading enzymes can work outside ideal laboratory conditions.",
  "N","def","https://www.latimes.com/tips/"),
 (14,"NBC News","National Tips Desk","National broadcast","Direct tips inbox","tips@nbcuni.com",
  "National youth-science story: AI-guided enzyme research for plastic waste",
  "National TV target; best after PET Lab has a strong local feature, visual assets, and verified results.",
  "Hi NBC News Tips Desk,",
  "This is a national youth-science story about two student founders trying to move a promising biotechnology concept closer to real-world environmental conditions.",
  "N","def","https://www.nbclosangeles.com/news/national-international/usda-accidentally-fired-officials-working-on-bird-flu-trying-to-rehire-them/3635703/"),
 (15,"ABC News","News Tips","National broadcast","Direct tips inbox","news.tips@abc.com",
  "ABC News story idea: Student founders use AI to tackle plastic pollution",
  "Large national audience; send only once the pitch includes a concise visual demo and independent expert voice.",
  "Hi ABC News Tips Team,",
  "The strongest national angle is the combination of young founders, accessible AI visuals, protein engineering, and a worldwide plastic-waste challenge.",
  "N","def","https://abcnews.go.com/US/tip-share-abc-news/story?id=61304290"),
 (16,"ABC News Live","Story Ideas Desk","National streaming news","Direct story-ideas inbox","ABCNewsLiveStoryIdeas@disney.com",
  "Visual story idea for ABC News Live: Students apply AI to enzyme design",
  "A visual streaming-news format that may be open to a short founder interview and demonstration.",
  "Hi ABC News Live Story Ideas Team,",
  "We can provide a concise, visual explanation of how a computer model compares possible protein mutations before researchers decide what should be tested in a lab.",
  "N","def","https://abcnews.go.com/US/viewers-voice-story-share-abc-news-live/story?id=70748073"),
 (17,"CBS News Philadelphia","News Tips","National/local broadcast","Direct tips inbox","tips@cbs.com",
  "CBS story tip: Student-led AI lab studies plastic-degrading enzymes",
  "A public CBS tips address that may route nationally relevant youth-science stories.",
  "Hi CBS News Tips Team,",
  "We are sharing a youth-science story that may be relevant beyond our home region because it explains a practical use of AI in biotechnology without treating early research as a finished solution.",
  "N","def","https://www.cbsnews.com/philadelphia/news-tip/"),
 (18,"CBS News Boston (WBZ)","News Tips","Regional broadcast","Direct tips inbox","newstips@wbztv.com",
  "Biotech youth story: Students use AI to study enzyme stability",
  "Boston has a strong biotech audience; the research angle may resonate even though the founders are in California.",
  "Hi WBZ News Desk,",
  "Although PET Lab is based in California, Boston audiences may find the biotechnology angle relevant: we are studying how computational methods can help prioritize enzyme mutations for difficult operating conditions.",
  "A","def","https://www.cbsnews.com/boston/about/"),
 (19,"The Washington Post","Secure Tips / Lockbox","National newspaper","Public tips email","lockbox@washpost.com",
  "Science and education pitch: Student founders build an AI enzyme lab",
  "Prestigious national target; use after independent experts and substantial evidence are ready.",
  "Hi Washington Post Newsroom,",
  "The broader story is about how accessible computational tools are changing who can participate in serious scientific research, while laboratory validation and careful claims remain essential.",
  "N","def","https://www.washingtonpost.com/information/2023/01/01/submit-an-anonymous-news-tip/"),
 (20,"Newsweek","Newsroom","National news magazine","Direct newsroom","newsroom@newsweek.com",
  "Youth innovation story: AI, protein engineering, and plastic pollution",
  "Broad national platform with interest in youth, innovation, AI, and environment stories.",
  "Hi Newsweek Newsroom,",
  "PET Lab is a student-led effort that brings together several timely subjects - AI, biotechnology, youth innovation, and plastic pollution - in one understandable research story.",
  "N","def","https://www.newsweek.com/contact"),
 (21,"ProPublica","General Editorial Contact","Investigative nonprofit","Indirect public contact - ask for routing","feedback@propublica.org",
  "Routing request: Student research, plastic waste, and real-world deployment claims",
  "Not a natural promotional target; only pitch a documented public-interest angle and ask for routing.",
  "Hi ProPublica Team,",
  "This is not a request for promotional coverage. We are asking whether there is an appropriate reporter for a transparent story about the difference between promising plastic-degrading enzymes, model predictions, and the evidence required for real-world deployment.",
  "N","route","https://www.propublica.org/contact"),
 (22,"Popular Science","Editorial Team","Science media","Direct editorial inbox","editorial@popsci.com",
  "Popular Science pitch: How students are using AI to redesign enzymes",
  "Strong fit for accessible explanations of AI, biology, environmental technology, and student invention.",
  "Hi Popular Science Editorial Team,",
  "The story can explain a real scientific workflow in accessible terms: how researchers represent protein mutations as data, predict stability, and decide what deserves experimental testing.",
  "S","def","https://www.popsci.com/contact/"),
 (23,"Science / AAAS","News from Science Tip Line","Science media","Direct science-news tips inbox","science_news@aaas.org",
  "News tip: Student-led computational study of enzyme stability conditions",
  "High-credibility science outlet; pitch only with methods, verified results, limitations, and independent context.",
  "Hi News from Science Team,",
  "We are sharing an early-stage computational biology project that may be relevant as a student-research or methods story, particularly because we are trying to model environmental conditions rather than reporting a laboratory-ready solution.",
  "S","def","https://www.science.org/content/page/got-tip"),
 (24,"WIRED","Kara Platoni, Science Editor","Science and technology media","Public editor email","kara_platoni@wired.com",
  "WIRED science pitch: What AI can - and cannot - predict about plastic enzymes",
  "Excellent intersection of AI, biology, climate technology, and the culture of student-led research.",
  "Hi Kara,",
  "The most interesting WIRED angle may be the boundary between computation and physical evidence: AI can prioritize mutations, but wet-lab testing still determines whether an enzyme truly works under real conditions.",
  "S","def","https://www.wired.com/story/how-to-pitch-stories-to-wired"),
 (25,"TechCrunch","Tim De Chant, Climate Reporter","Climate technology media","Public journalist email","tim.dechant@techcrunch.com",
  "Climate-tech pitch: Student lab applies AI to plastic-degrading enzymes",
  "Highly relevant for climate-tech framing, especially if PET Lab develops a clear pathway from research to deployment.",
  "Hi Tim,",
  "PET Lab is currently a research effort rather than a finished commercial product, but the deployment question is central: can computationally selected mutations help enzymes tolerate the conditions found in wastewater and industrial systems?",
  "C","def","https://techcrunch.com/about-techcrunch/"),
 (26,"The Verge","Tips Desk","Technology media","Direct tips inbox","tips@theverge.com",
  "Technology story tip: Student lab uses AI for protein engineering",
  "Useful for a technology-meets-science angle, particularly with clear visuals and a critical view of AI claims.",
  "Hi The Verge Tips Team,",
  "The technology story is not simply that we use AI; it is how model outputs are translated into biological hypotheses and where those predictions can fail without experimental validation.",
  "N","def","https://www.theverge.com/c/tech/22579076/how-to-tip-the-verge-email-signal-and-more"),
 (27,"Live Science","Editors","Science media","Direct editorial inbox","ls-editors@futurenet.com",
  "Live Science pitch: Can AI help plastic-degrading enzymes survive harsh conditions?",
  "Good fit for an accessible explainer about enzymes, mutation stability, and environmental constraints.",
  "Hi Live Science Editors,",
  "A useful reader question is whether a plastic-degrading enzyme that looks promising in one setting can still function when temperature, pH, salts, or other conditions change.",
  "S","def","https://www.livescience.com/how-to-pitch-live-science"),
 (28,"The Guardian","Science Desk","International science news","Direct science desk","science@theguardian.com",
  "Science pitch: Student researchers test the limits of AI-designed plastic enzymes",
  "Global audience and strong science/environment coverage; emphasize relevance beyond the United States.",
  "Hi Guardian Science Desk,",
  "The international relevance is the same everywhere plastic waste is processed: an enzyme must work in messy, changing environments, not only under ideal laboratory conditions.",
  "N","def","https://manage.theguardian.com/help-centre/article/contact-a-journalist-or-editorial-desk"),
 (29,"Futurism","Tips Desk","Technology and science media","Direct tips inbox","tips@futurism.com",
  "Futurism pitch: Student founders use AI to explore better plastic-degrading enzymes",
  "Future-facing AI and biotechnology outlet; keep the pitch evidence-based and avoid hype.",
  "Hi Futurism Tips Team,",
  "This is a forward-looking biotechnology story, but we want to present it responsibly: computational predictions are a way to narrow experiments, not proof that an optimized enzyme already works at scale.",
  "N","def","https://futurism.com/contact"),
 (30,"GEN - Genetic Engineering & Biotechnology News","Julianna LeMieux, Senior Science Writer","Biotechnology trade media","Public editorial email","julianna.lemieux@sagepub.com",
  "GEN story idea: Student-led computational enzyme engineering project",
  "Direct audience of biotechnology professionals; emphasize technical methods, validation plan, and limitations.",
  "Hi Julianna,",
  "For a biotechnology audience, we can discuss the model features, mutation-ranking workflow, condition variables, and the experimental validation that would be needed before making performance claims.",
  "S","def","https://www.genengnews.com/editorial-guidelines/"),
 (31,"Canary Media","Tips Desk","Climate solutions media","Direct tips inbox","tips@canarymedia.com",
  "Climate-solutions pitch: AI-guided enzymes for difficult waste-system conditions",
  "Strong fit for climate solutions and practical deployment questions in water and waste infrastructure.",
  "Hi Canary Media Tips Team,",
  "The solutions angle is the gap between enzyme discovery and infrastructure: treatment systems involve variable temperatures, chemistry, flow, and contaminants that computational screening could help researchers anticipate.",
  "C","def","https://www.canarymedia.com/contact"),
 (32,"Mongabay","Story Tips","Environmental news","Direct story-tips inbox","storytips@mongabay.com",
  "Environmental story tip: Student lab studies enzymes for real-world plastic waste",
  "International environmental readership; frame around plastic pollution, water systems, and evidence needed for impact.",
  "Hi Mongabay Story Tips Team,",
  "PET Lab focuses on a key environmental challenge: turning laboratory enzyme research into approaches that could eventually be tested under the complicated conditions found in water and waste systems.",
  "C","def","https://news.mongabay.com/submissions/"),
 (33,"CleanTechnica","Tips Desk","Clean technology media","Direct tips inbox","tips@cleantechnica.com",
  "Clean-tech story idea: Students bring AI to plastic-degrading enzyme research",
  "Clean-tech audience; explain the technology pathway without claiming commercial readiness.",
  "Hi CleanTechnica Tips Team,",
  "This is an early clean-technology research story with a practical question at its center: how can models help select enzyme changes that deserve testing under industrially relevant conditions?",
  "C","def","https://cleantechnica.com/2020/08/12/the-jackrabbit-version-2-0-e-bike-makes-riding-electric-easy-affordable/"),
 (34,"IEEE Spectrum","Michael Koziol, News Manager","Engineering and technology media","Public editor email","m.koziol@ieee.org",
  "IEEE Spectrum pitch: Engineering an AI pipeline for enzyme mutation screening",
  "Engineering readership; strong fit for model design, data pipelines, and translation from prediction to testing.",
  "Hi Michael,",
  "The engineering story is the end-to-end pipeline: combining protein, mutation, and environmental features to rank candidates while keeping uncertainty and experimental verification visible.",
  "S","def","https://spectrum.ieee.org/about"),
 (35,"STAT","Damian Garde, Biotech Reporter","Biotechnology news","Public journalist email","damian.garde@statnews.com",
  "Biotech pitch: Student founders build a condition-aware enzyme model",
  "Top biotech target; only send with rigorous results, expert context, and a clear reason the work matters now.",
  "Hi Damian,",
  "We know STAT applies a high bar to biotech claims. We would present PET Lab as an early computational research project, share the methods and limitations, and distinguish model performance from laboratory evidence.",
  "S","def","https://www.statnews.com/staff/damian-garde/"),
 (36,"Scientific American","Science Quickly Podcast","Science media","Public program email","sciencequickly@sciam.com",
  "Science Quickly idea: What AI really contributes to enzyme design",
  "Potential audio explainer about AI-guided biology, enzymes, and why real-world conditions matter.",
  "Hi Science Quickly Team,",
  "This could work as a short, accessible conversation about what AI contributes to protein engineering, what it cannot establish on its own, and why environmental conditions complicate plastic-degrading enzymes.",
  "S","def","https://www.scientificamerican.com/syndication/itunes-science-quickly.xml"),
 (37,"Grist","Media Team","Climate and environmental media","Indirect public contact - ask for routing","media@grist.org",
  "Routing request: Student climate-science project on plastic-degrading enzymes",
  "Excellent editorial fit, but this listed address is indirect; ask to be routed to a solutions or plastics reporter.",
  "Hi Grist Media Team,",
  "Could you please route this to the appropriate climate solutions or plastics reporter? PET Lab is a student-led research project studying how AI might help identify enzyme mutations for changing waste-system conditions.",
  "C","route","https://grist.org/contact/"),
 (38,"Ars Technica","Dan Goodin, Senior Security Editor","Technology media","Indirect public journalist email - ask for routing","dan.goodin@arstechnica.com",
  "Routing request for Ars: Student AI and protein-engineering research story",
  "Not his primary beat; use only as a polite routing request to an Ars science or AI editor.",
  "Hi Dan,",
  "We realize this is outside your main beat, so we are writing only to ask whether you can point us to the right Ars editor for a student-led AI and computational biology story.",
  "N","route","https://arstechnica.com/information-technology/2018/07/ongoing-scam-is-still-stoking-concerns-dell-customer-data-was-breached/"),
 (39,"Gizmodo","Tips Desk","Technology and science media","Public listing - recheck before sending","tips@gizmodomedia.com",
  "Science-tech tip: Student lab applies AI to plastic-degrading enzymes",
  "Potential science/technology coverage, but the public listing is older; verify that the inbox is still active.",
  "Hi Gizmodo Tips Team,",
  "The project is a concrete example of AI moving from software into physical science, with an important caveat: predictions only become meaningful when they are independently tested.",
  "N","def","https://gizmodo.com/facebook-wanted-us-to-kill-this-investigative-tool-1826620111"),
 (40,"EurekAlert!","Webmaster / General Contact","Science distribution service","Indirect public contact - ask for process","webmaster@eurekalert.org",
  "Question about eligibility for a student-led research announcement",
  "EurekAlert usually distributes eligible institutional releases; ask about the correct submission route or eligibility.",
  "Hi EurekAlert Team,",
  "We understand EurekAlert is primarily a science-news distribution service rather than a standard newsroom. Could you advise whether a student-led laboratory working with academic mentors is eligible to submit an announcement, or which institutional route we should use?",
  "S","eureka","https://www.eurekalert.org/contact"),
 (41,"TechCrunch","General Tips","Technology media","Direct tips inbox","tips@techcrunch.com",
  "TechCrunch tip: Student-led AI lab explores enzyme engineering for plastic waste",
  "General alternative to a named reporter; use only one TechCrunch address in the same outreach wave.",
  "Hi TechCrunch Tips Team,",
  "PET Lab is an early research initiative at the intersection of AI, biotechnology, and climate technology, with a goal of making candidate selection more informed before laboratory testing.",
  "N","def","https://techcrunch.com/about-techcrunch/"),
 (42,"TechCrunch","Russell Brandom, AI Editor","AI and technology media","Public editor email","russell.brandom@techcrunch.com",
  "AI pitch: Student researchers use machine learning to rank enzyme mutations",
  "AI-specific alternative; focus on responsible model use and the limits of computational prediction.",
  "Hi Russell,",
  "The AI angle is how machine-learning systems can rank biological candidates while still requiring domain expertise, uncertainty checks, and wet-lab validation.",
  "N","def","https://techcrunch.com/about-techcrunch/"),
 (43,"The Verge","Technology Pitches","Technology media","Direct pitches inbox","techpitches@theverge.com",
  "Technology pitch: Inside a student-built AI pipeline for protein mutations",
  "More targeted than the general tips inbox for a reported technology feature; do not contact both at once.",
  "Hi The Verge Technology Editors,",
  "A possible feature would follow the full path from protein sequence data to a ranked mutation, then show why computational confidence is not the same as biological proof.",
  "N","def","https://www.theverge.com/pages/how-to-pitch-the-verge"),
 (44,"WIRED","Ideas Desk","Technology ideas","Direct pitches inbox","ideas@wired.com",
  "WIRED Ideas pitch: AI is changing who can attempt serious biology",
  "Useful for a broader argument about AI expanding access to research while raising questions about evidence and expertise.",
  "Hi WIRED Ideas Editors,",
  "PET Lab could serve as a concrete case study in a broader idea: AI and open scientific tools are lowering barriers to sophisticated research, but credibility still depends on transparent methods, expert review, and experimental evidence.",
  "N","def","https://www.wired.com/story/how-to-pitch-stories-to-wired"),
 (45,"WIRED","Opinion Desk","Technology opinion","Direct opinion inbox","opinion@wired.com",
  "Opinion pitch: AI can widen access to biology - if we keep the evidence bar high",
  "Use only for a clearly argued, evidence-based essay - not a request for a profile.",
  "Hi WIRED Opinion Editors,",
  "We would like to propose an evidence-based first-person essay about using AI as student researchers: the tools expand what young teams can attempt, but they do not replace laboratory validation, mentorship, or scientific restraint.",
  "N","opinion","https://www.wired.com/story/how-to-pitch-stories-to-wired"),
 (46,"IEEE Spectrum","Harry Goldstein, Editor in Chief","Engineering and technology media","Public editor email","h.goldstein@ieee.org",
  "Engineering story idea: Student-built system screens enzyme mutations",
  "Senior editorial routing option; use one IEEE contact per wave and keep the pitch engineering-focused.",
  "Hi Harry,",
  "PET Lab may be relevant as an engineering education and computational systems story: students are integrating heterogeneous biological features into a reproducible screening workflow.",
  "S","def","https://spectrum.ieee.org/about"),
 (47,"IEEE Spectrum","Emily Waltz, Senior Editor","Engineering and biotechnology media","Public editor email","e.waltz@ieee.org",
  "Biotech engineering pitch: AI-guided screening for plastic enzyme stability",
  "Relevant to biotechnology and engineering; send instead of, not alongside, other IEEE addresses.",
  "Hi Emily,",
  "We can show how a protein-engineering question becomes an engineering pipeline involving data quality, feature construction, model evaluation, uncertainty, and experimental next steps.",
  "S","def","https://spectrum.ieee.org/about"),
 (48,"KTVU FOX 2","Kayla Galloway, Digital Journalist","Local digital news","Public journalist email - ask for routing","kayla.galloway@fox.com",
  "Digital local story: Bay Area student lab combines AI, biology, and sustainability",
  "Digital and social-friendly local route; use instead of the KTVU tips desk, not at the same time.",
  "Hi Kayla,",
  "The story is well suited to digital coverage because we can provide short video, clear model graphics, founder interviews, and a concise explanation of the science.",
  "A","def","https://www.ktvu.com/person/g/kayla-galloway"),
 (49,"KTVU FOX 2","Amber Lee, Reporter","Local broadcast","Public journalist email - ask for routing","Amber.Lee@Fox.com",
  "Bay Area youth-science story idea: AI-assisted enzyme research",
  "Another local journalist route; contact only if her recent coverage suggests a fit and do not duplicate the KTVU blast.",
  "Hi Amber,",
  "We are reaching out with a local youth-science story that can be told through both people and visuals, while remaining careful about the difference between model predictions and laboratory results.",
  "A","def","https://www.ktvu.com/person/l/amber-lee"),
 (50,"NBC Bay Area","Community Team","Local community outreach","Public community contact - ask for routing","nbcbayareacommunity@nbcuni.com",
  "Bay Area community story: Student founders build an environmental research lab",
  "Strong community/youth angle; ask whether the team can route the story to editorial or community programming.",
  "Hi NBC Bay Area Community Team,",
  "PET Lab is also a community youth-achievement story: two Bay Area student founders are building a collaborative research effort around AI, biology, and plastic pollution.",
  "A","route","https://www.nbcbayarea.com/community/"),
]

# ============================================================ PAGE 1: title
pdf.add_page()
pdf.ln(22)
pdf.set_font("S", "B", 30); pdf.set_text_color(*TEAL)
pdf.multi_cell(0, 13, "PET Lab", align="C", **NX)
pdf.multi_cell(0, 13, "Media Outreach Kit", align="C", **NX)
pdf.ln(3)
pdf.set_font("S", "", 13); pdf.set_text_color(*TEAL)
pdf.multi_cell(0, 7, "50 public email contacts + 50 tailored, copy-ready pitches", align="C", **NX)
pdf.ln(5)
by = pdf.get_y(); pdf.set_fill_color(*TEAL); pdf.rect(62, by, PW - 124, 6.5, "F")
pdf.ln(13)
pdf.set_font("S", "B", 11.5); pdf.set_text_color(*INK)
pdf.multi_cell(0, 6, "Prepared for Ayush Iyer & Abhinav Iyer, Co-Founders, Predictive Enzyme Technology Laboratory", align="C", **NX)
pdf.ln(1)
pdf.set_font("S", "", 9.5); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 5, "Research focus: AI-assisted protein engineering for plastic-degrading enzymes under changing real-world conditions", align="C", **NX)
pdf.ln(6)
bx = 26; bw = PW - 52; by = pdf.get_y(); bh = 48
pdf.set_fill_color(*LIGHTBG); pdf.rect(bx, by, bw, bh, "F")
pdf.set_xy(bx + 5, by + 4)
pdf.set_font("S", "B", 10.5); pdf.set_text_color(*INK)
pdf.cell(0, 5, "Important before sending"); pdf.ln(7)
for b in ["Replace every bracketed placeholder with verified information.",
          "Send individually in small waves. Never BCC all 50 or contact several people at the same outlet simultaneously.",
          "Attach or link a one-page fact sheet, one short demo, 2-3 strong visuals, and only results you can document.",
          "Use cautious language: \"researching,\" \"aims to,\" and \"may help\" - not \"solves plastic pollution.\""]:
    pdf.set_x(bx + 5)
    bullet(b, x=bx + 5, width=bw - 8)
pdf.set_y(by + bh + 6)
pdf.set_font("S", "", 9); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 5, "Verification date: July 21, 2026", align="C", **NX)

# ============================================================ PAGE 2: how to
pdf.add_page()
H1("How to use this kit")
leadpara("Start local, then scale.", "A strong Bay Area feature gives national and science outlets a credible external reference. The recommended first wave is contacts 1-10, followed by the most relevant science or industry contact for your strongest verified result.")
leadpara("One outlet, one contact, one wave.", "Several major organizations appear more than once because they publish different newsroom, desk, or journalist emails. Choose only the single best address at that organization, wait several business days, and then consider one polite follow-up.")
leadpara("Lead with a story, not a resume.", "The story is that Bay Area students are building a sustained interdisciplinary lab and using AI to study why plastic-degrading enzymes may fail when real-world environmental conditions change.")
H2("Prepare these assets first")
for b in ["A one-page fact sheet with founders, location, research question, current stage, mentors, and verified milestones.",
          "A 30-60 second video showing the model workflow, protein visualization, and founders explaining the problem.",
          "Two high-resolution horizontal photos and one vertical photo with permission to publish.",
          "One independent mentor or expert who is willing to be contacted for context.",
          "A simple evidence table separating computational results, external validation, and future wet-lab plans."]:
    bullet(b)
H2("Fill these placeholders before copying any email")
pdf.ln(0.5)
ph_rows = [
    ("[PET Lab website]", "Public website or media page"),
    ("[Your email]", "Professional PET Lab or school-appropriate email (Ayush)"),
    ("[Abhinav email]", "abhinaviyer555@gmail.com  (co-founder - already filled in signatures)"),
    ("[Your phone]", "Optional professional contact number"),
    ("[Verified result]", "One result with method, sample size, metric, and limitation"),
    ("[Independent expert]", "Mentor or outside researcher available for comment"),
    ("[Local connection]", "City, school, or Bay Area connection you are comfortable making public"),
]
for i, (k, v) in enumerate(ph_rows):
    h = 7.2; x = M; w1 = 46; y = pdf.get_y()
    if i % 2 == 0:
        pdf.set_fill_color(*ROWALT); pdf.rect(x, y, PW - 2 * M, h, "F")
    pdf.set_xy(x + 2, y + 1.6); pdf.set_font("S", "B", 8.6); pdf.set_text_color(*INK); pdf.cell(w1, 4, k)
    pdf.set_xy(x + w1 + 2, y + 1.6); pdf.set_font("S", "", 8.6); pdf.set_text_color(*INK); pdf.cell(0, 4, v)
    pdf.set_y(y + h)
H2("Source note")
pdf.set_x(M); pdf.set_font("S", "I", 8); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.3, "All addresses below were located on publicly accessible professional pages associated with the listed outlet and checked on July 21, 2026. A public email is not a guarantee that the inbox is active or that a pitch will be accepted. Entries marked indirect or recheck should be used as routing requests, not treated as confirmed editorial pitch channels.", **NX)

# ============================================================ PAGES 3-4: index
def index_header():
    x = M; y = pdf.get_y(); h = 7
    pdf.set_fill_color(*TEAL); pdf.rect(x, y, PW - 2 * M, h, "F")
    pdf.set_text_color(*WHITE); pdf.set_font("S", "B", 8)
    pdf.set_xy(x + 1, y + 1.7); pdf.cell(10, 4, "#")
    pdf.set_xy(x + 11, y + 1.7); pdf.cell(66, 4, "Outlet / contact")
    pdf.set_xy(x + 77, y + 1.7); pdf.cell(66, 4, "Email")
    pdf.set_xy(x + 143, y + 1.7); pdf.cell(40, 4, "Route")
    pdf.set_y(y + h + 0.6)


def index_row(n, outlet, role, email, route, fill):
    x = M; wnum = 11; wout = 66; wem = 66; wrt = PW - 2 * M - wnum - wout - wem
    ln_name = count_lines(outlet, wout, 7.8, "B")
    ln_role = count_lines(role, wout, 7.3, "")
    ln_em = count_lines(email, wem, 7.5, "")
    ln_rt = count_lines(route, wrt, 7.5, "")
    lh = 3.6
    h = max(ln_name + ln_role, ln_em, ln_rt) * lh + 2.2
    y = pdf.get_y()
    if fill:
        pdf.set_fill_color(*ROWALT); pdf.rect(x, y, PW - 2 * M, h, "F")
    pdf.set_xy(x + 1, y + 1.4); pdf.set_font("S", "B", 7.8); pdf.set_text_color(*TEAL); pdf.cell(wnum, lh, str(n))
    pdf.set_xy(x + wnum, y + 1.4); pdf.set_font("S", "B", 7.8); pdf.set_text_color(*INK)
    pdf.multi_cell(wout, lh, outlet, **NX)
    pdf.set_x(x + wnum); pdf.set_font("S", "", 7.3); pdf.set_text_color(*GREY)
    pdf.multi_cell(wout, lh, role, **NX)
    pdf.set_xy(x + wnum + wout, y + 1.4); pdf.set_font("S", "", 7.5); pdf.set_text_color(*INK)
    pdf.multi_cell(wem, lh, email, **NX)
    pdf.set_xy(x + wnum + wout + wem, y + 1.4); pdf.set_font("S", "", 7.5); pdf.set_text_color(*INK)
    pdf.multi_cell(wrt, lh, route, **NX)
    pdf.set_y(y + h)


pdf.add_page()
H1("Quick index: 50 public email contacts")
pdf.set_x(M); pdf.set_font("S", "", 8); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.3, "Ranked for PET Lab relevance. Duplicate organizations represent distinct public desks or journalists; use only one contact per organization in a given wave.", **NX)
pdf.ln(1.5)
index_header()
for i, p in enumerate(PITCHES):
    if pdf.get_y() > 250:
        pdf.add_page(); index_header()
    index_row(p[0], p[1], p[2], p[5], p[4], i % 2 == 0)

# ============================================================ PAGES 5+: pitches
def para(t, size=9.0):
    pdf.set_x(M); pdf.set_font("S", "", size); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.3, t, **NX); pdf.ln(0.9)


def card(p):
    n, outlet, role, cat, route, email, subj, why, greet, hook, tmpl, closer, src = p
    if pdf.get_y() > 140:
        pdf.add_page()
    pdf.ln(1.0)
    pdf.set_x(M); pdf.set_font("S", "B", 11.5); pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 6.0, "%02d   %s" % (n, outlet), **NX)
    pdf.set_x(M); pdf.set_font("S", "I", 8.0); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.2, "%s   |   %s   |   %s" % (role, cat, route), **NX)
    pdf.ln(0.4)
    kv("EMAIL", email, TEAL)
    kv("WHY THIS CONTACT", why)
    kv("SUBJECT", subj)
    pdf.set_x(M); pdf.set_font("S", "B", 8.4); pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 4.4, "EMAIL DRAFT", **NX); pdf.ln(0.4)
    para(greet)
    para(hook)
    para(make_body(tmpl, closer))
    pdf.set_font("S", "", 9.0); pdf.set_text_color(*INK)
    for line in SIG:
        pdf.set_x(M); pdf.multi_cell(0, 4.2, line, **NX)
    pdf.ln(1.0)
    pdf.set_x(M); pdf.set_font("S", "I", 7.7); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.1, "SOURCE   " + src, **NX)
    pdf.ln(1.4); pdf.set_draw_color(*RULE); y = pdf.get_y()
    pdf.line(M, y, PW - M, y); pdf.ln(2.4)


pdf.add_page()
H1("Tailored copy-ready pitches")
pdf.set_x(M); pdf.set_font("S", "", 8); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.3, "Copy the subject and body, replace placeholders, personalize one sentence, and verify the address again immediately before sending.", **NX)
pdf.ln(1)
for p in PITCHES:
    card(p)

# ============================================================ last page: sequence
pdf.add_page()
H1("Recommended sending sequence")
H2("Wave 1 - Local proof")
P("Send to one contact each at CBS Bay Area, KTVU, NBC Bay Area, ABC7, KQED, the Chronicle, the Standard, San Jose Spotlight, Pleasanton Weekly, and CalMatters.")
H2("Wave 2 - Specialist credibility")
P("Choose 5-8 outlets whose beat matches your strongest verified evidence: Popular Science, Science/AAAS, GEN, STAT, IEEE Spectrum, Canary Media, Mongabay, or Live Science.")
H2("Wave 3 - National scale")
P("After you have a local article, broadcast clip, expert quote, or substantial verified milestone, approach national television and major national newspapers.")
leadpara("Follow-up rule:", "Send one brief follow-up after 4-7 business days. Add one genuinely new item - a new result, video, independent expert, event, or local milestone. Do not repeatedly resend the same pitch.")
leadpara("Suggested follow-up subject:", "Follow-up: Bay Area student lab using AI for plastic-degrading enzymes")
bx = M; bw = PW - 2 * M; by = pdf.get_y(); bh = 52
pdf.set_fill_color(*LIGHTBG); pdf.rect(bx, by, bw, bh, "F")
pdf.set_xy(bx + 4, by + 3)
pdf.set_font("S", "", 9.2); pdf.set_text_color(*INK)
old = pdf.l_margin; pdf.set_left_margin(bx + 4)
for line in [
    "Hi [Name/Newsroom],", "",
    "I wanted to follow up once on the PET Lab story idea below. Since our first email, we have added [one new, verifiable development]. We can provide a concise fact sheet, visuals, and an interview, and we will be transparent about the current limitations of the research.",
    "", "If this is not a fit, no response is necessary. Thank you for considering it.",
    "", "Best,", "Ayush Iyer & Abhinav Iyer"]:
    if line == "":
        pdf.ln(2.2)
    else:
        pdf.set_x(bx + 4); pdf.multi_cell(bw - 8, 4.6, line, **NX)
pdf.set_left_margin(old)
pdf.set_y(by + bh + 5)
pdf.set_x(M); pdf.set_font("S", "I", 8); pdf.set_text_color(*GREY)
pdf.multi_cell(0, 4.3, "This kit is an outreach aid, not legal or public-relations advice. Avoid sharing private school records, home addresses, unpublished confidential research, or personal information about minors. Obtain permission before naming mentors, institutions, or collaborators.", **NX)

pdf.output(OUT)
print("wrote", OUT, "pages:", pdf.page_no())
