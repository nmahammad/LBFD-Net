import turtle
import math

# ------------------------------------------------------
# Helper: draw a heart of a given size
# ------------------------------------------------------
def draw_heart(t, size, color):
    t.color(color)
    t.begin_fill()
    t.left(140)
    t.forward(size)
    t.circle(size * -0.5, 200)
    t.left(120)
    t.circle(size * -0.5, 200)
    t.forward(size)
    t.end_fill()
    t.setheading(0)

# ------------------------------------------------------
# Setup screen + turtle
# ------------------------------------------------------
screen = turtle.Screen()
screen.bgcolor("white")
t = turtle.Turtle()
t.speed(3)

# ------------------------------------------------------
# Draw big heart
# ------------------------------------------------------
t.penup()
t.goto(0, -150)
t.pendown()
draw_heart(t, 200, "red")

# ------------------------------------------------------
# Draw small heart inside
# ------------------------------------------------------
t.penup()
t.goto(0, -80)
t.pendown()
draw_heart(t, 100, "pink")

# ------------------------------------------------------
# Write text on left
# ------------------------------------------------------
t.penup()
t.goto(-200, 50)
t.color("purple")
t.write("Girl with cool PJ", align="center", font=("Arial", 16, "bold"))

# ------------------------------------------------------
# Write text on right
# ------------------------------------------------------
t.goto(200, 50)
t.color("white")
t.write("Guy with - 6 eggs", align="center", font=("Arial", 16, "bold"))

# ------------------------------------------------------
# Finish
# ------------------------------------------------------
t.hideturtle()
turtle.done()
