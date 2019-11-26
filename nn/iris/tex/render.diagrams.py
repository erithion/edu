import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}

pgm = daft.PGM()

# input layer
pgm.add_node("input_0", r"$1$", 1, 6, plot_params=s_color)
pgm.add_node("input_1", r"$x_1$", 1, 5, plot_params=s_color)
pgm.add_node("dot1", r"", 1, 4.3, fixed=True)
pgm.add_node("dot2", r"", 1, 4, fixed=True)
pgm.add_node("dot3", r"", 1, 3.7, fixed=True)
pgm.add_node("input_4", r"$x_4$", 1, 3, plot_params=s_color)

# hidden layer
pgm.add_node(x=3, y=6,    plot_params=s_color,   node="z_0",               content=r"$1$")
pgm.add_node(x=3, y=5,    plot_params=s_color,   node="z_1",  aspect=2.1,  content=r"$z_{1}^{(1)} \mid a_{1}^{(1)}$")
pgm.add_node(x=3, y=4.3,                         node="dot4", fixed=True,  content=r"")
pgm.add_node(x=3, y=4,                           node="dot5", fixed=True,  content=r"")
pgm.add_node(x=3, y=3.7,                         node="dot6", fixed=True,  content=r"")
pgm.add_node(x=3, y=3,    plot_params=s_color,   node="z_8",  aspect=2.1,  content=r"$z_{8}^{(1)} \mid a_{8}^{(1)}$")

# input <-> hidden edges
pgm.add_edge("input_0", "z_1", label=r"$w_{1,0}^{(1)}$", xoffset=0.8, yoffset=-0.1, label_params={"rotation": -25})
pgm.add_edge("input_1", "z_1", label=r"$w_{1,1}^{(1)}$", xoffset=0.6, yoffset=0.2)
pgm.add_edge("input_4", "z_1", label=r"$w_{1,4}^{(1)}$", xoffset=0.9, yoffset=1.25, label_params={"rotation": 45})

pgm.add_edge("input_0", "z_8")
pgm.add_edge("input_1", "z_8")
pgm.add_edge("input_4", "z_8")

# output layer

pgm.add_node(x=5, y=5.5,    plot_params=s_color,   node="zz_1",  aspect=2.1,  content=r"$z_{1}^{(2)} \mid a_{1}^{(2)}$")
pgm.add_node(x=5, y=4.8,                           node="dot7", fixed=True,  content=r"")
pgm.add_node(x=5, y=4.5,                           node="dot8", fixed=True,  content=r"")
pgm.add_node(x=5, y=4.2,                           node="dot9", fixed=True,  content=r"")
pgm.add_node(x=5, y=3.5,    plot_params=s_color,   node="zz_3",  aspect=2.1,  content=r"$z_{3}^{(2)} \mid a_{3}^{(2)}$")

# hidden <-> output edges
pgm.add_edge("z_0", "zz_1", label=r"$w_{1,0}^{(2)}$", xoffset=0.8, yoffset=0.05,   label_params={"rotation": -15})
pgm.add_edge("z_1", "zz_1", label=r"$w_{1,1}^{(2)}$", xoffset=0.3, yoffset=0.3,   label_params={"rotation": 15})
pgm.add_edge("z_8", "zz_1", label=r"$w_{1,8}^{(2)}$", xoffset=0.9, yoffset=1.55, label_params={"rotation": 50})

pgm.add_edge("z_0", "zz_3")
pgm.add_edge("z_1", "zz_3")
pgm.add_edge("z_8", "zz_3")

# cost function
pgm.add_node(x=7, y=4.5,    plot_params=s_color,   node="cost",  content=r"C")

pgm.add_edge("zz_1", "cost")
pgm.add_edge("zz_3", "cost")

# Render and save.
pgm.render()
#pgm.savefig("exoplanets.pdf")
pgm.savefig("iris.nn.png", dpi=150)






# w2_C
pgm = daft.PGM()

# input layer
pgm.add_node("input_0", r"$1$", 1, 6, plot_params=s_color)
pgm.add_node("input_1", r"$x_1$", 1, 5, plot_params=s_color)
pgm.add_node("dot1", r"", 1, 4.3, fixed=True)
pgm.add_node("dot2", r"", 1, 4, fixed=True)
pgm.add_node("dot3", r"", 1, 3.7, fixed=True)
pgm.add_node("input_4", r"$x_4$", 1, 3, plot_params=s_color)

# hidden layer
pgm.add_node(x=3, y=6,    plot_params=s_color,   node="z_0",               content=r"$1$")
pgm.add_node(x=3, y=5.6,                         node="dot7", fixed=True,  content=r"")
pgm.add_node(x=3, y=5.3,                           node="dot8", fixed=True,  content=r"")
pgm.add_node(x=3, y=5.0,                         node="dot9", fixed=True,  content=r"")
pgm.add_node(x=3, y=4.5,    plot_params=s_color,   node="z_1",  aspect=2.1,  content=r"$z_{j}^{(1)} \mid a_{j}^{(1)}$")
pgm.add_node(x=3, y=4.0,                         node="dot4", fixed=True,  content=r"")
pgm.add_node(x=3, y=3.7,                           node="dot5", fixed=True,  content=r"")
pgm.add_node(x=3, y=3.4,                         node="dot6", fixed=True,  content=r"")
pgm.add_node(x=3, y=3,    plot_params=s_color,   node="z_8",  aspect=2.1,  content=r"$z_{8}^{(1)} \mid a_{8}^{(1)}$")

# output layer

pgm.add_node(x=5, y=5.6,                         node="dot10", fixed=True,  content=r"")
pgm.add_node(x=5, y=5.3,                           node="dot11", fixed=True,  content=r"")
pgm.add_node(x=5, y=5.0,                         node="dot12", fixed=True,  content=r"")
pgm.add_node(x=5, y=4.5,    plot_params=s_color,   node="zz_1",  aspect=2.1,  content=r"$\mathbf{z_{i}^{(2)} \mid a_{i}^{(2)}}$")
pgm.add_node(x=5, y=4.0,                         node="dot13", fixed=True,  content=r"")
pgm.add_node(x=5, y=3.7,                           node="dot14", fixed=True,  content=r"")
pgm.add_node(x=5, y=3.4,                         node="dot15", fixed=True,  content=r"")

pgm.add_edge("z_1", "zz_1", label=r"$\mathbf{w_{i,j}^{(2)}}$", yoffset=0.2)

# cost function
pgm.add_node(x=7, y=4.5,    plot_params=s_color,   node="cost",  content=r"$\mathbf{C}$")

pgm.add_edge("zz_1", "cost")

# Render and save.
pgm.render()
#pgm.savefig("exoplanets.pdf")
pgm.savefig("iris.nn.w2.png", dpi=150)











pgm = daft.PGM()

# input layer
pgm.add_node("input_0", r"$1$", 1, 6, plot_params=s_color)
pgm.add_node(x=1, y=5.6,                         node="il1", fixed=True,  content=r"")
pgm.add_node(x=1, y=5.3,                           node="il2", fixed=True,  content=r"")
pgm.add_node(x=1, y=5.0,                         node="il3", fixed=True,  content=r"")
pgm.add_node("input_1", r"$x_j$", 1, 4.5, plot_params=s_color)
pgm.add_node(x=1, y=4.0,                         node="il4", fixed=True,  content=r"")
pgm.add_node(x=1, y=3.7,                           node="il5", fixed=True,  content=r"")
pgm.add_node(x=1, y=3.4,                         node="il6", fixed=True,  content=r"")
pgm.add_node("input_4", r"$x_4$", 1, 3, plot_params=s_color)

# hidden layer
pgm.add_node(x=3, y=6,    plot_params=s_color,   node="z_0",               content=r"$1$")
pgm.add_node(x=3, y=5.6,                         node="hl1", fixed=True,  content=r"")
pgm.add_node(x=3, y=5.3,                           node="hl2", fixed=True,  content=r"")
pgm.add_node(x=3, y=5.0,                         node="hl3", fixed=True,  content=r"")
pgm.add_node(x=3, y=4.5,    plot_params=s_color,   node="z_1",  aspect=2.1,  content=r"$\mathbf{z_{i}^{(1)} \mid a_{i}^{(1)}}$")
pgm.add_node(x=3, y=4.0,                         node="hl4", fixed=True,  content=r"")
pgm.add_node(x=3, y=3.7,                           node="hl5", fixed=True,  content=r"")
pgm.add_node(x=3, y=3.4,                         node="hl6", fixed=True,  content=r"")
pgm.add_node(x=3, y=3,    plot_params=s_color,   node="z_8",  aspect=2.1,  content=r"$z_{8}^{(1)} \mid a_{8}^{(1)}$")

# input <-> hidden edges
pgm.add_edge("input_1", "z_1", label=r"$\mathbf{w_{i,j}^{(1)}}$", xoffset=0.6, yoffset=0.2)

# output layer

pgm.add_node(x=5, y=5.5,    plot_params=s_color,   node="zz_1",  aspect=2.1,  content=r"$\mathbf{z_{1}^{(2)} \mid a_{1}^{(2)}}$")
pgm.add_node(x=5, y=4.5,    plot_params=s_color,   node="zz_2",  aspect=2.1,  content=r"$\mathbf{z_{2}^{(2)} \mid a_{2}^{(2)}}$")
pgm.add_node(x=5, y=3.5,    plot_params=s_color,   node="zz_3",  aspect=2.1,  content=r"$\mathbf{z_{3}^{(2)} \mid a_{3}^{(2)}}$")

# hidden <-> output edges
pgm.add_edge("z_1", "zz_1")
pgm.add_edge("z_1", "zz_2")
pgm.add_edge("z_1", "zz_3")

# cost function
pgm.add_node(x=7, y=4.5,    plot_params=s_color,   node="cost",  content=r"$\mathbf{C}$")

pgm.add_edge("zz_1", "cost")
pgm.add_edge("zz_2", "cost")
pgm.add_edge("zz_3", "cost")

# Render and save.
pgm.render()
#pgm.savefig("exoplanets.pdf")
pgm.savefig("iris.nn.w1.png", dpi=150)





