"""Generic speaker-role words, per language.

These matter only for lower-case labels. An ALL-CAPS label is a label in every
language, so it needs no list; lower case is where guessing goes wrong, because a
colon shows up in ordinary dialogue too ("First vote: Jeff", "Eu falei: nao vou").
Cutting on the colon alone eats a word mid-sentence, so lower case is restricted to
this closed list of generic roles and never matches a name.
"""

ROLES = {
    "en": (
        "man woman men women boy girl child children kid kids narrator announcer "
        "reporter anchor officer policeman policewoman cop doctor nurse judge both "
        "all everyone crowd audience driver soldier captain waiter waitress clerk "
        "operator receptionist radio tv television phone telephone computer machine "
        "singer chorus choir voice voices male female teacher student guard pilot "
        "dispatcher intercom speaker"
    ),
    "pt": (
        "homem mulher homens mulheres menino menina garoto garota crianca criança "
        "criancas crianças policial medico médico medica médica enfermeiro enfermeira "
        "reporter repórter locutor locutora narrador narradora apresentador "
        "apresentadora juiz juiza juíza ambos ambas todos todas amigo amiga multidao "
        "multidão publico público plateia trabalhador trabalhadores professor "
        "professora motorista soldado capitao capitão doutor doutora senhor senhora "
        "garcom garçom recepcionista operador operadora radio rádio televisao "
        "televisão telefone computador maquina máquina anuncio anúncio comercial "
        "cantor cantora coro voz vozes guarda piloto"
    ),
    "es": (
        "hombre mujer hombres mujeres nino niño nina niña chico chica policia policía "
        "medico médico enfermera juez ambos todos todas amigo multitud publico público "
        "conductor soldado capitan capitán camarero recepcionista operador radio "
        "television televisión telefono teléfono computadora maquina máquina cantante "
        "coro voz voces maestro guardia piloto narrador locutor reportero"
    ),
    "fr": (
        "homme femme hommes femmes garcon garçon fille enfant enfants policier medecin "
        "médecin infirmiere infirmière juge tous toutes ami foule public conducteur "
        "soldat capitaine serveur receptionniste opérateur radio television télévision "
        "telephone téléphone ordinateur machine chanteur choeur chœur voix professeur "
        "garde pilote narrateur annonceur journaliste"
    ),
    "de": (
        "mann frau männer frauen junge mädchen kind kinder erzähler sprecher "
        "polizist arzt ärztin krankenschwester richter beide alle menge publikum "
        "zuschauer fahrer soldat hauptmann kellner kellnerin empfangschef betreiber "
        "funk radio fernsehen telefon computer maschine sänger chor stimme stimmen "
        "lehrer student wächter pilot"
    ),
    "it": (
        "uomo donna uomini donne ragazzo ragazza bambino bambina bambini narratore "
        "annunciatore poliziotto medico infermiere giudice entrambi tutti folla "
        "pubblico conducente autista soldato capitano cameriere barista operatore "
        "centralinista radio televisione telefono computer macchina cantante coro "
        "voce voci insegnante studente guardia pilota"
    ),
}


def words_for(languages):
    """Union of the role words for the given language codes."""
    out = []
    for lang in languages:
        for word in ROLES.get(lang, "").split():
            if word not in out:
                out.append(word)
    return out
