texfile = open("GeneratedTex.txt", "a")

r = 0
filecounter = [i for i in range(1, 452)]
filecounter.sort(key=str)
imgcounter = 0

for i in range(0, 451):
    imgfile = f"./nbimg/file ({filecounter[r]}).jpg"
    piefile = f"./nbimg/pie-{imgcounter}.jpg"
    peakfile = f"./nbimg/peak-{imgcounter}.jpg"
    r += 1
    imgcounter += 1
    text = """
    \\begin{center}
    \includegraphics[width=\\textwidth]{%s}
    \end{center}

    \\begin{center}
    \includegraphics[width=250mm]{%s}
    \end{center}

    \\begin{center}
    \includegraphics[width=250mm]{%s}
    \end{center}
    """ % (imgfile, piefile, peakfile)
    texfile.write(text)
    texfile.write("\n")


texfile.close()