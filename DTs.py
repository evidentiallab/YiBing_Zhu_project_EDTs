from manimlib import *
from os import * 
from numpy.lib.function_base import iterable
from sklearn.datasets import load_iris
import numpy as np
BUFF_CIRCLE= 0.5

def split_iris_data(choose_dim: list = [0, 1, 2]):
    if not iterable(choose_dim):
        raise ValueError("choose_dim must be an iterable object!")
    if not isinstance(choose_dim, list):
        choose_dim = list(choose_dim)

    x, y = load_iris(return_X_y=True)
    split_data = {0:[],1:[],2:[]}
    for x, y in zip(x,y):
        split_data[y].append(x[choose_dim])
    return split_data

class DTs(Scene):
    def construct(self):
        headline = Text("Welcome to ......",font = "Consolas", font_size = 90 )
        subheadline = Text("Made by Mathisbing from UTSEUS SHANGHAI UNIVERSITY", font = "Arial", font_size = 50, t2c={"Mathisbing": BLUE, "UTSEUS":ORANGE})
        VGroup(headline, subheadline).arrange(DOWN, buff = 1)
        self.play(
                Write(headline),
                FadeIn(subheadline, UP)
                )
        self.play(
                FadeOut(headline),
                FadeOut(subheadline, shift = DOWN)
                )
        
        headline_2 = Text(
                '''
                What is the Decision Tree?
                ''',
                font = "Arial",
                t2f = {"Decision":"Consolas","Tree":"Consolas"}
                )
        headline_2.set_color_by_t2g(t2g={
           "What is the Decision Tree?":[YELLOW,BLUE]
        })
        self.play(Write(headline_2))
        self.play(FadeOut(headline_2))
        self.wait()

        headline_3 = Text(
                '''
                A decision tree is a tree structure,
                in which each internal node represents
                a judgment on an attribute,each branch
                represents the output of a judgment
                result,and finally each leaf node
                represents a classification result.
                ''',
                font = "Arial"
                )
        circle = Circle(radius =0.4,stroke_width=2)
        circle2 = Circle(radius =0.4,stroke_width=2,color = GREEN)
        circle3 = Circle(radius =0.4,stroke_width=2,color = BLUE)
        circle4 = circle2.copy().set_color(PURPLE)
        circle5 = circle2.copy().set_color(PINK)
        circle6 = circle3.copy().set_color(WHITE)
        circle7 = circle3.copy().set_color(ORANGE)
        circle8 = circle.copy()
        
        circle2.next_to(circle,DR,buff =BUFF_CIRCLE)
        circle3.next_to(circle,DL,buff =BUFF_CIRCLE)
        circle4.next_to(circle2,DR,buff=BUFF_CIRCLE)
        circle5.next_to(circle2,DL,buff=BUFF_CIRCLE)
        circle6.next_to(circle5,DR,buff=BUFF_CIRCLE)
        circle7.next_to(circle5,DL,buff=BUFF_CIRCLE)
        circle8.next_to(circle7,DL,buff=BUFF_CIRCLE)

        line1 = Line(circle.get_center(),circle2.get_center(),buff = 0.4)
        line2 = Line(circle.get_center(),circle3.get_center(),buff = 0.4)
        line3 = Line(circle2.get_center(),circle5.get_center(),buff = 0.4)
        line4 = Line(circle2.get_center(),circle4.get_center(),buff = 0.4)
        line5 = Line(circle5.get_center(),circle6.get_center(),buff = 0.4)
        line6 = Line(circle5.get_center(),circle7.get_center(),buff = 0.4)
        line7 = Line(circle7.get_center(),circle8.get_center(),buff = 0.4)
        

        vg1 = VGroup(circle,circle2,line1,line2,circle3)
        vg1.add(circle4,circle5,circle6,circle7,circle8)
        vg1.add(line3,line4,line5,line6,line7)
        vg1.move_to(UP)
        vg_d = VGroup(headline_3, vg1).arrange(RIGHT, buff = 1)
        vg_d.scale(0.8)
        self.play(FadeIn(vg_d))
        self.wait()



        label = Text("root node")
        label_2 = Text("node")
        label_3 = Text("leaf node")
        label_4 = Text("branch")
        self.play(FadeOut(headline_3)) 
        self.play(FadeIn(label),ShowCreationThenDestructionAround(circle))
        self.wait()
        self.play(FadeOut(label))
        self.play(FadeIn(label_2),ShowCreationThenDestructionAround(circle4))
        self.wait()
        self.play(FadeOut(label_2))
        self.play(FadeIn(label_3),ShowCreationThenDestructionAround(circle8))
        self.wait()
        self.play(FadeOut(label_3))
        self.play(FadeIn(label_4),ShowCreationThenDestructionAround(VGroup(circle3,line2)))
        self.wait()
        self.play(FadeOut(label_4),FadeOut(vg1))
    
        
        headline_4 = Text(
                '''
                Using IRIS data , as a example 
                We choose the three most discriminative features to establish coordinates
                sepal length (cm),petal length (cm),petal width (cm)
                ''',
                font = "Arial",
                font_size = 32
                )
        self.play(
                FadeIn(headline_4),
                )
        self.wait(2)
        self.play(
                FadeOut(headline_4)
                )
        self.wait()


        split_data = split_iris_data([0, 2, 3])
        colors = [BLUE, RED, GREEN]
        dot_clouds = [
                DotCloud(
                    points = np.array(split_data[y]),
                    )
                    for y in split_data
                ]
        axes = ThreeDAxes()
        frame : CameraFrame =self.camera.frame
        frame.scale(1.3)

        params = {
                "width":4,
                "height":2,
                "fill_color":PURPLE_E,
                "fill_opacity":0.75
                }
        classify_data = {
                'decision_node_1':"petal length<=2.45",
                'decision_node_2':"petal width <= 1.75",
                'decision_node_3':"setosa",
                'decision_node_4':"versicolor",
                'decision_node_5':"virginica"
                }
        cir_text = Text(classify_data['decision_node_1'],font='mysh',font_size=38)
        ob_test = RoundedRectangle(**params)
        vg2 = VGroup(cir_text,ob_test)
        ob_test3 = ob_test.copy()
        ob_test2 = ob_test.copy()
        ob_test4 = ob_test.copy()
        ob_test5 = ob_test.copy()
        cir_text2 = Text(classify_data['decision_node_2'],font="mysh",font_size=38)
        vg3 = VGroup(ob_test2,cir_text2)
        self.play(ShowCreation(vg2))
        self.play(vg2.animate.move_to(UP*3))
        vg3.move_to(LEFT*3)
        line_test = Line(vg2.get_bottom(),vg3.get_top())
        vg3.add(line_test)
        self.play(ShowCreation(vg3))
        cir_text3 = Text(classify_data['decision_node_3'],font="mysh",font_size=38)
        vg4 = VGroup(ob_test3,cir_text3)
        vg4.move_to(RIGHT*3)
        line_test1 = Line(vg2.get_bottom(),vg4.get_top())
        vg4.add(line_test1)
        self.play(ShowCreation(vg4))
        cir_text3 = Text(classify_data['decision_node_4'],font="mysh",font_size=38)
        vg5 = VGroup(ob_test4,cir_text3)
        vg5.move_to(LEFT*5+DOWN*3)
        line_test2 = Line(vg3.get_bottom(),vg5.get_top())
        vg5.add(line_test2)
        self.play(ShowCreation(vg5))
        cir_text4 = Text(classify_data['decision_node_5'],font="mysh",font_size=38)
        vg6 = VGroup(ob_test5,cir_text4)
        vg6.move_to(RIGHT+DOWN*3)
        line_test3 = Line(vg3.get_bottom(),vg6.get_top())
        vg6.add(line_test3)
        self.play(ShowCreation(vg6))
        self.wait(2)


        frame : CameraFrame =self.camera.frame
        frame.set_euler_angles(
                theta = 110* DEGREES,

                phi= 80* DEGREES,
                )

        self.wait()
        self.add(axes)
        self.add(*dot_clouds)
        self.wait(2)
        dot_clouds_c=[DotCloud(
                points = np.array(split_data[y]),
                color = colors[y]
                )for y in split_data]
        surface = ParametricSurface(
            uv_func=lambda u, v : [u, v, 1.75],
            u_range=(-7, 7),
            v_range=(-7, 7),
            color=BLUE,
            opacity=0.1
        )
        surface_2 = ParametricSurface(
            uv_func=lambda u, v : [u, 2.45, v],
            u_range=(-7, 7),
            v_range=(-7, 7),
            color=BLUE,
            opacity=0.1
        )


        self.wait(1)
        self.add(surface_2)
        self.play(
                FadeIn(dot_clouds_c[0]),
                )
        self.wait(1)
        self.add(surface)
        self.play(
                FadeIn(dot_clouds_c[1]),
                )
        self.wait(1)
        self.play(
                FadeIn(dot_clouds_c[2])
               )
        self.wait(1)

if __name__ == "__main__":
    system("manimgl {} {} -f  ".format(__file__,os.path.basename(__file__).rstrip(".py")))
