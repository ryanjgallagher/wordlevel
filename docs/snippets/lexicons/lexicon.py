# --8<-- [start:labmt]
import wordlevel as wl

labmt = wl.lex.labMT("english")
# --8<-- [end:labmt]


# --8<-- [start:socialsent]
import wordlevel as wl

social_sent = wl.lex.SocialSent("2000")
# --8<-- [end:socialsent]


# --8<-- [start:revision]
import wordlevel as wl

latest_loader = wl.LexiconLoader(revision="latest")
labmt = latest_loader.labMT("english")

# The commit the lexicon was loaded at, to load exactly the same version elsewhere
pinned_loader = wl.LexiconLoader(revision=labmt.revision)
pinned_labmt = pinned_loader.labMT("english")
# --8<-- [end:revision]


# --8<-- [start:update-cached]
import wordlevel as wl

wl.lex.update_cached()
# --8<-- [end:update-cached]
