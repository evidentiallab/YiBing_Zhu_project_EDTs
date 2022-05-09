from manimlib import *
from os import *
import numpy as np


class EDTs(Scene):
    def set_frame(self, frame_scale:int) -> None:
        frame : CameraFrame =self.camera.frame
        frame.scale(frame_scale)


    def first_frame(self):
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
        self.play(*[ShowCreation(mob) for mob in [reclabel_text[1],reclabel_text[2],reclabel_text[3]]])
        self.play(ShowCreation(reclabel_text[4]))
        self.play(ShowCreation(reclabel_text[5]))
        self.play(ShowCreation(reclabel_text[6]))
        self.play(ShowCreation(reclabel_text[7]))
        self.play(*[ShowCreation(mo) for mo in [reclabel_text[8],reclabel_text[9]]])
        self.add(*line_all)
        self.wait(2)
        self.clear()
        
       
    def second_frame(self):
        self.set_frame(0.5)
        bit_map = ImageMobject(
                filename = './images/Table.png',
                height=6,
                opacity=0.9
                )
        bit_map.scale(0.8)
        frame : CameraFrame =self.camera.frame
        frame.scale(2)
        bit_map.move_to(UP*4+LEFT*4)
        self.add(bit_map)
        frame_bit_map = Tex(r'\Theta_{j}\ :\ frame\ of\ discernment\\ A^{j}\ :\ set\ of\ evidential\ test\ attribute')
        frame_attribute = VGroup()
        frame_attribute.add(Tex(r'A^{(1)}\ =\ \{Ability\},\ \Theta_{1}\ = \ \{G,Y,B\},\ P(\Theta_{1})=\{\emptyset,G,Y,B,G\cup Y,G\cup B, Y\cup B, G\cup Y\cup B\}'))
        frame_attribute.add(Tex(r'Instance\quad e_{p}\quad BBA\quad on\quad the\quad attribute\quad A^{(j)}\quad is\quad denoted\quad as\quad m\{e_{p}\}(\Theta_{j})'))
        frame_attribute.add(Tex(r'Example:\quad m\{e_{1}\}(\Theta_1)\ :\ m\{e_{1}\}(G)=0.8\ and\ m\{e_{1}\}(B)=0.2'))
        frame_attribute[1].move_to(DOWN)
        frame_attribute[2].move_to(DOWN*2)
        frame_attribute.move_to(DOWN*3)
        self.play(
            ShowCreation(frame_attribute)
        )
        #self.add(*frame_attribute)
        #self.clear()
        cir  = VGroup(
            Circle(radius = 8).move_to(LEFT+DOWN),
            Circle(radius = 4, fill_opacity = 0.4, stroke_opacity = 0.4),
            Circle(radius = 4, stroke_color = BLUE, fill_opacity = 0.4, stroke_opacity = 0.4, fill_color =BLUE).move_to(LEFT*4),
            Circle(radius = 4, stroke_color = GREEN, fill_opacity = 0.4, stroke_opacity = 0.4, fill_color =GREEN).move_to(LEFT*2+DOWN*4)
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
        #self.add(cir)
        self.add(text)
        self.wait(5)
        self.clear()


    def third_frame(self):
        self.set_frame(0.5)
        formula1 = Tex('Test\ Attribute\ Selection\ :\ Map\_dis(D,A^{(j)})=\sum_{p,q=1,p<q}^{p,q=N}{|S_{D}(e_{p},e_{q})-S_{A^{(j)}}(e_{p},e_{q})|}').scale(0.5).move_to(UP*2)
        tog_tex_rec1 = VGroup(Tex('Ability'),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec2 = VGroup(Tex('Property'),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec3 = VGroup(Tex('Appearance'),Rectangle()).arrange(ORIGIN).scale(0.5)
        tog_tex_rec1.move_to(UP*3)
        
        text = VGroup(
            Tex("G",font_size=64),
            Tex("Y",font_size=64),
            Tex("B",font_size=64),
            Tex("G\cup Y",font_size=64),
            Tex("Y\cup B",font_size=64),
            Tex("B\cup G",font_size=64),
            Tex("G\cup Y\cup B",font_size=64)
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

        self.add(formula1)
        self.wait(2)
        self.add(partition_text[1:4])
        self.wait(2)
        self.play(FadeOut(formula1),FadeOut(partition_text[1:4]))        
        self.add(tog_tex_rec1)
        self.wait(2)
        self.add(arrow,text)
        self.wait(2)

        label_t = VGroup(
            Tex(r'a_{1}^{1}',font_size=64).scale(0.4).next_to(text[0],DOWN),
            Tex(r'a_{2}^{1}',font_size=64).scale(0.4).next_to(text[1],DOWN),
            Tex(r'a_{3}^{1}',font_size=64).scale(0.4).next_to(text[2],DOWN),
            Tex(r'a_{4}^{1}',font_size=64).scale(0.4).next_to(text[3],DOWN),
            Tex(r'a_{5}^{1}',font_size=64).scale(0.4).next_to(text[4],DOWN),
            Tex(r'a_{6}^{1}',font_size=64).scale(0.4).next_to(text[5],DOWN),
            Tex(r'a_{7}^{1}',font_size=64).scale(0.4).next_to(text[6],DOWN)
        )
        self.add(label_t)
        partition_text[0].move_to(DOWN)
        self.add( partition_text[0])
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
        self.wait(3)
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
       

        self.add(dot)
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
        self.wait(1)

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
        self.add(text2,arrow2)
        self.add(text[0:3],text[6])
        self.wait(3)
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
            Tex("Mu",font_size=64),
            Tex("L",font_size=64),
            Tex("Mo",font_size=64),
            Tex("L",font_size=64),
            Tex("Y",font_size=64),
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

        example_text.move_to(UP*4+RIGHT*5)    
        self.add(example_text)
        example_cir = VGroup(Circle(),Tex(r'e_{0}',font_size=64)).arrange(ORIGIN).scale(0.4).move_to(tog_tex_rec1.get_top())
        self.play(
            ApplyMethod(example_cir.move_to,text2[1].get_center()+DOWN),
            ApplyMethod(example_cir.copy().move_to,text3[0].get_center()+DOWN),
            run_time = 3
        ) 
        self.clear()

        example_for = VGroup(
                Tex(r'R(e_{0},e_{i})=1-\frac{1}{K}\sum_{j=1}^{n}{W_{j}.d_{BPA}(m\{e_{0}\}(\Theta_{j}),m\{e_{i}\}(\Theta_{j}))}'),
                Tex(r'W_{j}=Dis\_Cap(A^{j})'),
                Tex(r'K=\sum_{j=1}^{n}{Dis\_Cap(A^{j})}'),
                Tex(r'the\ matching\ coefficient\ related\ to\ e13\ and\ e5\ is\ respectively\,\ 0.87\ and\ 0.51')
        ).arrange(DOWN).move_to(UP*2)
        self.add(example_for)
        self.wait(2)
        example_for2 = VGroup(
                Tex(r'm(C)=\sum_{C_{i}\cap C_{j}...\cap C_{W}=C}{m_{1}(C_{i}).m_{1}(C_{j})...m_{n_{e_{0}}}}(C_{W})+\sum\ \tau{(C)}'),
                Tex(r'm\{e_{0}\}\{\phi,C_{1},C_{2},C_{3},C_{1}C_{2},C_{1}C_{3},C_{2}C_{3},C_{1}C_{2}C_{3}\}=\{0,0.7,0.24,0,0.03,0,0,0.03\}'),
                Tex(r'BetP(A)=\sum_{B\subset \Theta}{\frac{|A\cap B|}{|B|}.\frac{m(B)}{1-m(\phi)}}'),
                Tex(r'p\{e_{0}\}\{C_{1},C_{2},C_{3}=\{0.73,0.26,0.01\}\}')
        ).arrange(DOWN)
        example_for2.next_to(example_for,DOWN)
        self.add(example_for2)
        self.wait()


    def construct(self):
        self.first_frame()
        self.second_frame()
        self.third_frame()


if __name__ == "__main__":
    system(" manimgl {} {} -f ".format(__file__,os.path.basename(__file__).rstrip(".py")))
