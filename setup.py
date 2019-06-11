from distutils.core import setup
import py2exe
setup (windows = ['Interfacee.pyw'],
       options = { 'py2exe' : {'packages':['Tkinter']}})
