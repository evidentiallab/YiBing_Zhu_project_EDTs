from manimlib import *
from os import *
import numpy as np


class Frame3(Scene):
    def set_frame(self, frame_scale:int) -> None:
        frame : CameraFrame =self.camera.frame
        frame.scale(frame_scale)


    def first_frame(self):
        self.add_sound('./sounds/f3_1.m4a')
        headline_6 = Title("EDTs",fons_size=30)
        headline_6.move_to(UP*6)
        self.set_frame(2)
        self.add(headline_6)

        reclabel = [
                'data',
                'mass(attribute 1)',
                'mass(attribute 2)',
                'mass(attribute 3)',
                'new data',
                'attribute select',
                'that attribute as root',
                'stopping criteria', 
                'leaf node',
                'combination'
                ]
        rect_list =[]
        label_text =[]
        reclabel_text = []
        for i in iter(reclabel):
            rect_list.append(Rectangle(width=4,height=1))
            label_text.append(Text(i,font_size=30))
        for r_t, l_t in zip (rect_list,label_text):
            reclabel_text.append(VGroup(l_t,r_t))
        
        reclabel_text[0].move_to(UP*5)
        reclabel_text[1].move_to(UP*3+LEFT*6)
        reclabel_text[2].move_to(UP*3+RIGHT*6)
        reclabel_text[3].move_to(UP*3)
        reclabel_text[4].move_to(UP)
        reclabel_text[5].move_to(DOWN)
        reclabel_text[6].move_to(DOWN*3)
        reclabel_text[7].move_to(DOWN*5)
        reclabel_text[8].move_to(DOWN*7)
        reclabel_text[9].move_to(DOWN*7+RIGHT*6)

        line_all = []
        line_all.append(Line(start=reclabel_text[0].get_bottom(),end=reclabel_text[1],buff=0.4))
        line_all.append(Line(start=reclabel_text[0].get_bottom(),end=reclabel_text[2],buff=0.4))
        line_all.append(Line(start=label_text[0].get_bottom(),end=label_text[3],buff=0.4))
        line_all.append(Line(start=reclabel_text[1].get_bottom(),end=reclabel_text[4],buff=0.4))
        line_all.append(Line(start=reclabel_text[2].get_bottom(),end=reclabel_text[4],buff=0.4))
        line_all.append(Line(start=label_text[3].get_bottom(),end=label_text[4],buff=0.4))
        line_all.append(Line(start=label_text[4].get_bottom(),end=label_text[5],buff=0.4))
        line_all.append(Line(start=label_text[5].get_bottom(),end=label_text[6],buff=0.4))
        line_all.append(Line(start=label_text[6].get_bottom(),end=label_text[7],buff=0.4))
        line_all.append(Line(start=label_text[7].get_bottom(),end=label_text[8],buff=0.4))
        line_all.append(Line(start=reclabel_text[8].get_right(),end=reclabel_text[9].get_left(),buff=0.4))
        line_all.append(CurvedArrow(
                start_point=reclabel_text[7].get_right(),
                end_point=reclabel_text[5].get_right(),
                angle = TAU / 4,
                stroke_width=4,
                stroke_color = BLUE_A,
                tip_width_ratio = 2
                ))

        self.play(ShowCreation(reclabel_text[0]))
        self.wait(2)
        self.play(*[ShowCreation(mob) for mob in [reclabel_text[1],reclabel_text[2],reclabel_text[3]]])
        self.wait(2)
        self.play(ShowCreation(reclabel_text[4]))
        self.wait(2)
        self.play(ShowCreation(reclabel_text[5]))
        self.wait(2)
        self.play(ShowCreation(reclabel_text[6]))
        self.wait(2)
        self.play(ShowCreation(reclabel_text[7]))
        self.wait(2)
        self.play(*[ShowCreation(mo) for mo in [reclabel_text[8],reclabel_text[9]]])
        self.add(*line_all)
        self.wait(28)
        self.clear()
        
       
    def second_frame(self):
        self.add_sound('./sounds/f3_2.m4a')
        bit_map = ImageMobject(
                filename = './images/Table.png',
                height=6,
                opacity=0.9
                )
        # bit_map.scale(0.8)
        frame : CameraFrame =self.camera.frame
        frame.scale(2)
        bit_map.move_to(UP*4+LEFT*4)
        self.add(bit_map)
        frame_bit_map = Tex(r'\Theta_{j}\ :\ frame\ of\ discernment\\ A^{j}\ :\ set\ of\ evidential\ test\ attribute')
        frame_attribute = VGroup(
            Tex(r'A^{(j)}\ :\ the\ j^{th}\ attribute',r',\quad', r'\Theta_{j}\ : \ the\ frame\ of\ discernment\ of\ the\ j^{th}\ attribute'),
            Tex(r'For\ example:', r'A^{(1)}\ =\ \{Ability\}',r',', r'\Theta_{1}\ = \ \{G,Y,B\}'),
            Tex(r'P(\Theta_{1})=\{\emptyset,G,Y,B,G\cup Y,G\cup B, Y\cup B, G\cup Y\cup B\}'),
            Tex(r'Instance\quad e_{p}\quad BBA\quad on\quad the\quad attribute\quad A^{(j)}\quad is\quad denoted\quad as\quad m\{e_{p}\}(\Theta_{j})'),
            Tex(r'Example:\quad m\{e_{1}\}(\Theta_1)\ :\ m\{e_{1}\}(G)=0.8\ and\ m\{e_{1}\}(B)=0.2')
            ).arrange(DOWN)
        frame_attribute.move_to(DOWN*3)
        self.play(
            ShowCreation(frame_attribute[0])
        )
        self.play(
            TransformMatchingTex(frame_attribute[0].copy(), frame_attribute[1]),
            run_time=2 
        )
        self.play(
            ShowCreation(frame_attribute[2:])
        )

        cir  = VGroup(
            Circle(radius = 8).move_to(LEFT+DOWN),
            Circle(radius = 4, fill_opacity = 0.4, stroke_opacity = 0.4),
            Circle(radius = 4, stroke_color = BLUE, fill_opacity = 0.4, stroke_opacity = 0.4, fill_color =BLUE),
            Circle(radius = 4, stroke_color = GREEN, fill_opacity = 0.4, stroke_opacity = 0.4, fill_color =GREEN)
        )
        text = VGroup(
            Tex("G",font_size=64).move_to(LEFT*6),
            Tex("Y",font_size=64).move_to(RIGHT*2),
            Tex("B",font_size=64).move_to(LEFT*2+DOWN*5),
            Tex("G\cup Y",font_size=64).move_to(LEFT*2+UP),
            Tex("Y\cup B",font_size=64).move_to(DOWN*3),
            Tex("B\cup G",font_size=64).move_to(LEFT*4+DOWN*3),
            Tex("G\cup Y\cup B",font_size=64).move_to(LEFT*2+DOWN),
            Tex("\emptyset",font_size=64).move_to(RIGHT*4+UP*3),
            Tex(r'\Theta_{1}',font_size=64).next_to(cir[0],UP)
        )
        self.wait(2)
        tog_cir_text = VGroup(cir,text).scale(0.5).move_to(RIGHT*7+UP*3)
        self.play(
            *[ShowCreation(mob) for mob in cir]
        )
        self.play(
            cir[2].animate.shift(LEFT*2),
            ApplyMethod(cir[3].shift,LEFT*1+DOWN*2),
            run_time=2
        )
        self.add(text)
        self.wait(46)
        self.clear()


    def third_frame(self): 
        self.add_sound('./sounds/f3_3.m4a')
        color_dict = {"G" : BLUE, "Y" : ORANGE, "B" : RED, "Ability" : GREEN, "Property" : GREEN, "Appearance" : GREEN,
                        "Mu" : PURPLE,  "L" : YELLOW, "Mo" : MAROON,"Y" : GOLD}
        formula1 = Tex('Test\ Attribute\ Selection\ :\ Map\_dis(D,A^{(j)})=\sum_{p,q=1,p<q}^{p,q=N}{|S_{D}(e_{p},e_{q})-S_{A^{(j)}}(e_{p},e_{q})|}').scale(0.5).move_to(UP*2)
        tog_tex_rec1 = VGroup(Tex('Ability', tex_to_color_map=color_dict),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec2 = VGroup(Tex('Property', tex_to_color_map=color_dict),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec3 = VGroup(Tex('Appearance', tex_to_color_map=color_dict),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec1.move_to(UP*3)
        
        text = VGroup(
            Tex("G",font_size=64, tex_to_color_map=color_dict),
            Tex("Y",font_size=64, tex_to_color_map=color_dict),
            Tex("B",font_size=64, tex_to_color_map=color_dict),
            Tex("G\cup Y",font_size=64, tex_to_color_map=color_dict),
            Tex("Y\cup B",font_size=64, tex_to_color_map=color_dict),
            Tex("B\cup G",font_size=64, tex_to_color_map=color_dict),
            Tex("G\cup Y\cup B",font_size=64, tex_to_color_map=color_dict)
        ).arrange(RIGHT,buff=2.5).scale(0.5)

        arrow = VGroup()
        for i in range(7):
            arrow.add(Arrow(start=tog_tex_rec1.get_bottom(),end=text[i].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8,))
        
        partition_text = VGroup(
            Tex(r'max\{m\{e_{p}(\Theta_{j})\}\}=m\{e_{p}\}(a_{i}^{j})\quad and \quad m\{e_{p}\}(a_{i}^{j})\geqslant\alpha'),
            Tex(r'Map\_Dis(D,Ability)=6.87'),
            Tex(r'Map\_Dis(D,Appearance)=11.47'),
            Tex(r'Map\_Dis(D,Property)=8.64')
            ).arrange(DOWN).scale(0.5)

        # self.add(formula1)
        # self.wait(2)
        # self.add(partition_text[1:4])
        # self.wait(2)
        # self.play(FadeOut(formula1),FadeOut(partition_text[1:4]))        
        self.play(ShowCreation(tog_tex_rec1))
        self.wait(11)
        self.play(ShowCreation(arrow),
                ShowCreation(text))
        self.wait(20)

        label_t = VGroup(
            Tex(r'a_{1}^{1}',font_size=64).scale(0.4).next_to(text[0],DOWN),
            Tex(r'a_{2}^{1}',font_size=64).scale(0.4).next_to(text[1],DOWN),
            Tex(r'a_{3}^{1}',font_size=64).scale(0.4).next_to(text[2],DOWN),
            Tex(r'a_{4}^{1}',font_size=64).scale(0.4).next_to(text[3],DOWN),
            Tex(r'a_{5}^{1}',font_size=64).scale(0.4).next_to(text[4],DOWN),
            Tex(r'a_{6}^{1}',font_size=64).scale(0.4).next_to(text[5],DOWN),
            Tex(r'a_{7}^{1}',font_size=64).scale(0.4).next_to(text[6],DOWN)
        )
        self.play(ShowCreation(label_t))
        partition_text[0].move_to(DOWN)
        self.play(ShowCreation( partition_text[0]))
        self.wait(2)
        self.play(FadeOut(partition_text[0]))

        bit_map = ImageMobject(
                filename = './images/Table.png',
                height=6,
                opacity=0.9
                )
        bit_map.scale(0.8)
        formula2 = Tex(r'max\{m\{e_{1}\}(\Theta_{1})\}=max\{\ m\{e_{1}\}(G),\ m\{e_{1}\}(B)\ \}=0.8=max\{m\{e_{1}\}(G)\}=max\{m\{e_{1}\}(a_{1}^{1})\}').scale(0.5)
        formula2.move_to(DOWN*3) 
        self.add(bit_map)
        self.play(ShowCreation(formula2))
        self.wait(12)
        self.play(
            FadeOut(formula2),
            FadeOut(bit_map)
        )
        self.wait(1)
       
        dot = VGroup(
            VGroup(Circle(),Tex(r'e_{1}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{2}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{3}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{4}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{5}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{6}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{7}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top()),
            VGroup(Circle(),Tex(r'e_{8}')).arrange(ORIGIN).scale(2).move_to(tog_tex_rec1.get_top())
        )
       

        self.play(ShowCreation(dot))
        self.play(
            ApplyMethod(dot.scale, 0.2),
            run_time = 3
        )
        self.play(
            ApplyMethod(dot[0].move_to, text[0].get_center()+DOWN),
            ApplyMethod(dot[1].move_to, text[6].get_center()+DOWN),
            ApplyMethod(dot[2].move_to, text[0].get_center()+DOWN*2),
            ApplyMethod(dot[3].move_to, text[0].get_center()+DOWN*3),
            ApplyMethod(dot[4].move_to, text[1].get_center()+DOWN),
            ApplyMethod(dot[5].move_to, text[2].get_center()+DOWN),
            ApplyMethod(dot[6].move_to, text[2].get_center()+DOWN*2),
            ApplyMethod(dot[7].move_to, text[2].get_center()+DOWN*3),
            run_time = 3
        )

        self.play(
            FadeOut(label_t[3:6]),
            FadeOut(arrow[3:6]),
            FadeOut(text[3:6]),
            run_time=3
        )     
        self.wait(6)

        self.play(
            FadeOut(label_t[0:3]),
            FadeOut(label_t[6]),
            FadeOut(arrow[0:3]),
            FadeOut(arrow[6]),
            FadeOut(text[0:3]),
            FadeOut(text[6]),
            FadeOut(dot),
            run_time=3
        )
        
        
        text2 = VGroup(
            VGroup(Tex('Property'),Rectangle()).arrange(ORIGIN).scale(0.5),
            Tex(r'm\{e_{5}\}(\Theta)',font_size=64).scale(0.8).next_to(text[4],DOWN),
            VGroup(Tex('Appearance'),Rectangle()).arrange(ORIGIN).scale(0.5),
            VGroup(Tex('Appearance'),Rectangle()).arrange(ORIGIN).scale(0.5)
        ).arrange(RIGHT,buff=1)

        arrow2 = VGroup()
        for i in range(4):
            arrow2.add(Arrow(start=tog_tex_rec1.get_bottom(),end=text2[i].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8,))
        text[0].next_to(text2[0],UP)
        text[1].next_to(text2[1],UP)
        text[2].next_to(text2[2],UP)
        text[6].next_to(text2[3],UP)
        self.play(ShowCreation(text2),
        ShowCreation(arrow2))
        self.play(ShowCreation(text[0:3]),
        ShowCreation(text[6]))
        self.wait(1 )
        text3 = VGroup(
            Tex(r'm\{e_{13}\}(\Theta)',font_size=64).scale(0.7),
            Tex(r'm\{e_{4}\}(\Theta)',font_size=64).scale(0.7),
            Tex(r'm\{e_{78}\}(\Theta)',font_size=64).scale(0.7),
            Tex(r'm\{e_{6}\}(\Theta)',font_size=64).scale(0.7),
            Tex(r'm\{e_{2}\}(\Theta)',font_size=64).scale(0.7),
            Tex(r'm\{e_{9}\}(\Theta)',font_size=64).scale(0.7)
        ).arrange(RIGHT,buff=1).move_to(DOWN*4)

        self.set_frame(1.5)
        arrow3 = VGroup(
            Arrow(start=text2[0].get_bottom(),end=text3[0].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8),
            Arrow(start=text2[0].get_bottom(),end=text3[1].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8,)   
        )
        arrow4 = VGroup(
            Arrow(start=text2[2].get_bottom(),end=text3[2].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8),
            Arrow(start=text2[2].get_bottom(),end=text3[3].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8,)   
        )
        arrow5 = VGroup(
            Arrow(start=text2[3].get_bottom(),end=text3[4].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8),
            Arrow(start=text2[3].get_bottom(),end=text3[5].get_center(),fill_color=BLUE,fill_opacity=0.5,thickness=0.005, tip_width_ratio=8,)   
        )
        
        self.play(
            ShowCreation(text3),
            ShowCreation(arrow3), 
            ShowCreation(arrow4), 
            ShowCreation(arrow5)
            )

        text4 = VGroup(
            Tex("Mu",font_size=64, tex_to_color_map=color_dict),
            Tex("L",font_size=64, tex_to_color_map=color_dict),
            Tex("Mo",font_size=64, tex_to_color_map=color_dict),
            Tex("L",font_size=64, tex_to_color_map=color_dict),
            Tex("Y",font_size=64, tex_to_color_map=color_dict),
            Tex(r'\Theta_{2}',font_size=64),
        ).scale(0.5)
        for i in range(6):
            text4[i].next_to(text3[i],UP+LEFT)
        self.play(
            ShowCreation(text4)
            )
        self.wait(4)
        example_text = VGroup(
                Tex(r'm\{e_{0}\}\{(Ability)\}=(G:0.8,Y:0.2)'),
                Tex(r'm\{e_{0}\}\{(Appearance)\}=(Y:0.8,\Theta_{2}:0.2)'),
                Tex(r'm\{e_{0}\}\{(Property)\}=(Mu:0.7,Mo:0.3)')
        ).arrange(DOWN)
        example_name = Tex(r'assume\ the\ instance\ e_{0}\ to\ classify')
        example_name.move_to(UP*4+LEFT*5)
        example_text.move_to(UP*4+RIGHT*5.5)    
        self.add_sound('./sounds/f3_4_g.m4a')
        self.play(ShowCreation(example_text),
        ShowCreation( example_name))
        self.wait(8)
        example_cir = VGroup(Circle(),Tex(r'e_{0}',font_size=64)).arrange(ORIGIN).scale(0.4).move_to(tog_tex_rec1.get_top())
        self.play(
            ApplyMethod(example_cir.move_to,text2[1].get_center()+DOWN),
            ApplyMethod(example_cir.copy().move_to,text3[0].get_center()+DOWN),
            run_time = 3
        )
        self.play(
            ShowCreationThenDestructionAround(VGroup(text3[0],arrow3[0])),
            ShowCreationThenDestructionAround(VGroup(text2[1],arrow2[1])),
            run_time = 5
        )
        self.wait(3)
        self.clear()
        
    def fourth_frame(self, leaf_node=None):
        self.add_sound('./sounds/f3_5_0.m4a')
        example_cir = VGroup(Circle(),Tex(r'e_{0}',font_size=64)).arrange(ORIGIN).scale(0.4)
        leaf_node = [
            VGroup(Tex(r'm\{e_{13}\}(\Theta)',font_size=64).scale(0.7),example_cir.copy()).arrange(DOWN),
            VGroup(Tex(r'm\{e_{5}\}(\Theta)',font_size=64).scale(0.7),example_cir.copy()).arrange(DOWN)
            ]

        leaf_node[0].move_to(UP*4+LEFT*8)
        leaf_node[1].next_to(leaf_node[0],RIGHT*3)
        self.add(*leaf_node)
        
        e1 = VGroup(Tex(r'e_{1}:\quad'),
        Tex(r'\{G:0.8,B:0.2\}'),
        Tex(r'\{G:0.6,Y:0.4\}'),
        Tex(r'\{Mu:0.9,Mo:0.1\}'),
        Tex(r'\{C_{1}:0.8,C_{1}C_{2}:0.2\}')
        ).arrange(RIGHT)
        e3 = VGroup(Tex(r'e_{3}:\quad'),
        Tex(r'\{G:0.6,B:0.4\}'),
        Tex(r'\{Y:0.5,GY:0.4\}'),
        Tex(r'\{Mu:0.8,Mo:0.1,L:0.1\}'),
        Tex(r'\{C_{2}:0.6,C_{1}C_{2}C_{3}:0.2\}')
        ).arrange(RIGHT)
        e13 = VGroup(Tex(r'e_{13}:\quad'),
        Tex(r'\{G:0.7,B:0.3\}'),
        Tex(r'\{G:0.3,Y:0.45,GY:0.25\}'),
        Tex(r'\{Mu:0.85,Mo:0.1,L:0.05\}'),
        Tex(r'\{C_{1}:0.8,C_{1}C_{2}:0.2,C_{1}C_{2}C_{3}:0.1\}')
        ).arrange(RIGHT)

        e13.next_to(e3,DOWN*6)
        e1.next_to(e3,UP*6)

        arrow = CurvedArrow(
            start_point=leaf_node[0].get_left(),
            end_point=e1.get_left()+UP,
            angle=TAU / 16.  
        )
        arrow.set_color(color=RED)
        write_vg = VGroup(
            Tex(r'combine')
        )
        write_vg.next_to(e1[0],DOWN).shift(LEFT)
        self.add(arrow)
        self.play(
            *[TransformMatchingTex(
                leaf_node[0][0].copy(), b,
                path_arc=90 * DEGREES,
            ) for b in e3],
            run_time =2 
        )
        self.add(write_vg)
        self.wait(1)
        self.play(
            *[TransformMatchingTex(
                leaf_node[0][0].copy(), b,
                path_arc=90 * DEGREES,
            ) for b in e1],
            run_time =2 
        )
    
        line_vg = VGroup()
        for i in range(1,len(e1)):
            line_vg.add(Line(start=e1[i],end=e3[i]).set_color(BLUE))
        self.add(*line_vg)
        arrow_vg = VGroup()
        for i in range(1,len(e1)):
            arrow_vg.add(CurvedArrow(
            start_point=e3[i].get_center(),
            end_point=e13[i].get_top(),
            angle=TAU / 16.             # 45度
        ).set_color(ORANGE))
        self.add(*arrow_vg)

        self.play(
            *[TransformMatchingTex(
                leaf_node[0][0].copy(), b,
                path_arc=90 * DEGREES,
            ) for b in e13],
            run_time =2 
        )

        title = Text('combine the data in the leaf node ',gradient=[RED, ORANGE, YELLOW, GREEN, BLUE, BLUE_E, PURPLE]).next_to(leaf_node[1],UP*3)
        self.add(title)
        e13_cir = VGroup(Circle(),Tex(r'e_{13}',font_size=64)).arrange(ORIGIN).next_to(e13[0],DOWN*4+RIGHT*3)
        arr = Arrow(
            start=e13[0],
            end=e13_cir,
            fill_color=GREEN,
            fill_opacity=0.8,
            thickness=3,
            tip_width_ratio=8,
            tip_angle=PI / 6.
        )
        self.add(e13_cir,arr)
        

        example_cir_copy = VGroup(Circle(),Tex(r'e_{5}',font_size=64)).next_to(e13_cir,RIGHT*8)
        darr = DoubleArrow(
                start=e13_cir,
                end=example_cir_copy,
                color=BLUE,
                fill_color=BLUE
            )
        self.add(darr,example_cir_copy)
        relation = Text('How to combine the two leaf nodes ?',t2c={"How": BLUE, "combine": RED}).next_to(example_cir_copy, RIGHT*3)
        self.add(relation)
        self.wait(10)
        self.clear()

    def fifth_frame(self):
        self.add_sound('./sounds/f3_5.m4a')
        headline_7 = Title('Combine the two leaf nodes to get the result ').move_to(UP*6).scale(2)
        self.add(headline_7)
        self.wait(1)
        e0 = VGroup(Circle(),Tex(r'e_{0}',font_size=64))
        e5 = VGroup(Circle(),Tex(r'e_{5}',font_size=64))
        e13 = VGroup(Circle(),Tex(r'e_{13}',font_size=64)).next_to(e0, RIGHT*4)
        e13.move_to(UP*4)
        e0.next_to(e13,LEFT*8)
        self.add(e0,e13) 
        self.wait(1)
        doublearrow = DoubleArrow(
                start=e0,
                end=e13,
                color=BLUE,
                fill_color=BLUE
            )
        self.add(doublearrow)
        self.wait(1)
        relation =  VGroup(
            Text('calculate the relation between e0 and e13 ',t2c={"calculate": BLUE, "e0": RED, "e13": RED}),
            Text('we get R{e0,e13}=0.72',t2c={"R{e0,e13}": ORANGE}), 
            Text('it indicates the matching degree of the two nodes')
            ).arrange(DOWN)
        relation.next_to(doublearrow,DOWN*4)
        self.add(relation)
        self.wait(1)
        rel = VGroup(
            Text('R{e0,e13} x ',t2c={"R{e0,e13}": ORANGE}), 
            Text('R{e0,e5} x ',t2c={"R{e0,e5}": ORANGE}),
            Text('Result:'),
            Text('+')
        )
        e13_c = e13.copy().move_to(ORIGIN-UP+LEFT*4)
        rel[0].next_to(e13_c,LEFT)
        self.add(e13_c,rel[0])
        self.wait(1) 
        rel[3].next_to(rel[0],RIGHT*12)
        rel[1].next_to(rel[0],RIGHT*15)
        e5_c = e5.copy().next_to(rel[1],RIGHT)
        rel[2].next_to(rel[0],DOWN*6+LEFT*2)
        self.add(rel[1], e5_c, rel[2],rel[3])
        res_text = VGroup(
            Tex(r'm\{e_{0}\}\{\phi,C_{1},C_{2},C_{3},C_{1}C_{2},C_{1}C_{3},C_{2}C_{3},C_{1}C_{2}C_{3}\}=\{0,0.7,0.24,0,0.03,0,0,0.03\}')
        ).next_to(rel[2],RIGHT)
        self.add(res_text)
        self.wait(1)

    def construct(self):
        self.first_frame()
        # self.set_frame(0.5)
        # # self.wait(1)
        # self.second_frame()
        # self.set_frame(0.5)
        # # self.wait(1)
        # self.third_frame()
        # self.set_frame(1.3)
        # # self.wait(1)
        # self.fourth_frame()
        # self.fifth_frame()

if __name__ == "__main__":
    system(" manimgl {} {} -f -w ".format(__file__,os.path.basename(__file__).rstrip(".py")))
