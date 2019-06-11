from bbfreeze import Freezer

freezer = Freezer(distdir='dist')
freezer.addScript('interface.py', gui_only=True)
freezer()
