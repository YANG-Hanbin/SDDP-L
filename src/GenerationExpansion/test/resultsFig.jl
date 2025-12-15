## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the same instance with different cuts -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
algorithm = :SDDPL;
tightness = false;
for T in [10, 15]
    for num in [5, 10]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝色、橙色、绿色
        # 读取数据
        sddlpResultLC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-LC-$tightness.jld2")["sddpResults"][:solHistory]
        sddlpResultPLC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-ELC-$tightness.jld2")["sddpResults"][:solHistory]
        sddlpResultSMC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-ShrinkageLC-$tightness.jld2")["sddpResults"][:solHistory]

        # 处理 gap 数据
        sddlpResultLC.gap_float = parse.(Float64, replace.(sddlpResultLC.gap, "%" => "")) 
        sddlpResultPLC.gap_float = parse.(Float64, replace.(sddlpResultPLC.gap, "%" => ""))
        sddlpResultSMC.gap_float = parse.(Float64, replace.(sddlpResultSMC.gap, "%" => ""))

        # 统一数据格式
        df_LC = DataFrame(Iter=sddlpResultLC.iter, Time=sddlpResultLC.Time, LB=sddlpResultLC.LB, UB=sddlpResultLC.UB, Cut="LC")
        df_PLC = DataFrame(Iter=sddlpResultPLC.iter, Time=sddlpResultPLC.Time, LB=sddlpResultPLC.LB, UB=sddlpResultPLC.UB, Cut="PLC")
        df_SMC = DataFrame(Iter=sddlpResultSMC.iter, Time=sddlpResultSMC.Time, LB=sddlpResultSMC.LB, UB=sddlpResultSMC.UB, Cut="SMC")

        # 合并数据
        df = vcat(df_LC, df_PLC, df_SMC)

        # 绘制 Lower Bound (LB) 随时间变化
        df |> @vlplot(
            layer=[
                # Lower Bound (LB) 线，虚线
                {
                    :line,
                    transform=[{filter="datum.Time <= 300"}],
                    x={:Time, axis={title="Time (sec.)", titleFontSize=25, labelFontSize=25}},
                    y={:LB, axis={title="Bounds", titleFontSize=25, labelFontSize=25}},  
                    color={
                        :Cut, 
                        legend={title=nothing, orient="top", columns=3}, 
                        scale={domain=["LC", "PLC", "SMC"], range=colors}
                    },  
                    strokeDash={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=[[5, 3], [10, 2], [10, 5, 2, 5]]}
                    },  # LB 继续用虚线
                    shape={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=["circle", "diamond", "cross"]}
                    },  
                    strokeWidth={value=1}  # LB 细
                },
                # Upper Bound (UB) 线，实线 & 加粗
                {
                    :line,
                    transform=[{filter="datum.Time <= 300"}],
                    x=:Time,
                    y=:UB,  
                    color={
                        :Cut, 
                        legend={title=nothing, orient="top", columns=3}, 
                        scale={domain=["LC", "PLC", "SMC"], range=colors}
                    },  
                    strokeDash={value=[]},  # UB 改成实线
                    shape={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=["circle", "diamond", "cross"]}
                    },  
                    strokeWidth={value=1}  # UB 线加粗
                }
            ],
            width=500,
            height=350,
            config={ 
                axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
                legend={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman",
                    labelFontSize=25,  
                    symbolSize=150,    
                    symbolStrokeWidth=3  
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-bounds_Time_Period$T-Real$num.pdf")

        # 绘制 Lower Bound (LB) 随迭代次数变化
        df |> @vlplot(
            layer=[
                # Lower Bound (LB) 线，虚线
                {
                    :line,
                    transform=[{filter="datum.Iter <= 30"}],
                    x={:Iter, axis={title="Iteration", titleFontSize=25, labelFontSize=25}},
                    y={:LB, axis={title="Bounds", titleFontSize=25, labelFontSize=25}},  
                    color={
                        :Cut, 
                        legend={title=nothing, orient="top", columns=3}, 
                        scale={domain=["LC", "PLC", "SMC"], range=colors}
                    },  
                    strokeDash={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=[[5, 3], [10, 2], [10, 5, 2, 5]]}
                    },  # LB 继续用虚线
                    shape={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=["circle", "diamond", "cross"]}
                    },  
                    strokeWidth={value=1}  # LB 细
                },
                # Upper Bound (UB) 线，实线 & 加粗
                {
                    :line,
                    transform=[{filter="datum.Iter <= 30"}],
                    x=:Iter,
                    y=:UB,  
                    color={
                        :Cut, 
                        legend={title=nothing, orient="top", columns=3}, 
                        scale={domain=["LC", "PLC", "SMC"], range=colors}
                    },  
                    strokeDash={value=[]},  # UB 改成实线
                    shape={
                        :Cut, 
                        scale={domain=["LC", "PLC", "SMC"], range=["circle", "diamond", "cross"]}
                    },  
                    strokeWidth={value=1}  # UB 线加粗
                }
            ],
            width=500,
            height=350,
            config={ 
                axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
                legend={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman",
                    labelFontSize=25,  
                    symbolSize=150,    
                    symbolStrokeWidth=3  
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-bounds_Iter_Period$T-Real$num.pdf")
    end
end


algorithm = :SDDiP;
for T in [10, 15]
    for num in [5, 10]  
        # color setup
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        sddlpResultLC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-LC-$tightness.jld2")["sddpResults"][:solHistory]
        sddlpResultPLC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-ELC-$tightness.jld2")["sddpResults"][:solHistory]
        sddlpResultSMC = load("src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-ShrinkageLC-$tightness.jld2")["sddpResults"][:solHistory]

        sddlpResultLC.gap_float = parse.(Float64, replace.(sddlpResultLC.gap, "%" => "")) 
        sddlpResultPLC.gap_float = parse.(Float64, replace.(sddlpResultPLC.gap, "%" => ""))
        sddlpResultSMC.gap_float = parse.(Float64, replace.(sddlpResultSMC.gap, "%" => ""))

        df_LC_LB = DataFrame(Iter=sddlpResultLC.iter, Time=sddlpResultLC.Time, Bound=sddlpResultLC.LB ./ 10^3, Cut="LC", BoundType="Lower Bound")
        df_LC_UB = DataFrame(Iter=sddlpResultLC.iter, Time=sddlpResultLC.Time, Bound=sddlpResultLC.UB ./ 10^3, Cut="LC", BoundType="Upper Bound")

        df_PLC_LB = DataFrame(Iter=sddlpResultPLC.iter, Time=sddlpResultPLC.Time, Bound=sddlpResultPLC.LB ./ 10^3, Cut="PLC", BoundType="Lower Bound")
        df_PLC_UB = DataFrame(Iter=sddlpResultPLC.iter, Time=sddlpResultPLC.Time, Bound=sddlpResultPLC.UB ./ 10^3, Cut="PLC", BoundType="Upper Bound")

        df_SMC_LB = DataFrame(Iter=sddlpResultSMC.iter, Time=sddlpResultSMC.Time, Bound=sddlpResultSMC.LB ./ 10^3, Cut="SMC", BoundType="Lower Bound")
        df_SMC_UB = DataFrame(Iter=sddlpResultSMC.iter, Time=sddlpResultSMC.Time, Bound=sddlpResultSMC.UB ./ 10^3, Cut="SMC", BoundType="Upper Bound")

        df = vcat(df_LC_LB, df_LC_UB, df_PLC_LB, df_PLC_UB, df_SMC_LB, df_SMC_UB)


        df |> @vlplot(
            :line,
            transform=[{filter="datum.Time <= 50"}],
            x={:Time, axis={title="Time (sec.)", titleFontSize=25, labelFontSize=25}},
            y={:Bound, axis={title="Bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=3},  
                scale={domain=["LC", "PLC", "SMC"], range=colors}  
            },  
            strokeDash={
                :BoundType,  
                legend={title=nothing, orient="top", columns=2},
                scale={domain=["Lower Bound", "Upper Bound"], range=[[5, 3], [1, 0]]}  
            },  
            width=500,
            height=350,
            config={ 
                axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
                legend={
                    labelFont="Times New Roman", titleFont="Times New Roman",
                    labelFontSize=20, symbolSize=150, symbolStrokeWidth=3  
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-bounds_Time_Period$T-Real$num.pdf")

        #  Lower/upper Bounds vs Iter
        df |> @vlplot(
            :line,
            transform=[{filter="datum.Iter <= 20"}],
            x={:Iter, axis={title="Iteration", titleFontSize=25, labelFontSize=25}},
            y={:Bound, axis={title="Bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=3},  
                scale={domain=["LC", "PLC", "SMC"], range=colors}  
            },  
            strokeDash={
                :BoundType,  
                legend={title=nothing, orient="top", columns=2},
                scale={domain=["Lower Bound", "Upper Bound"], range=[[5, 3], [1, 0]]}  
            },  
            width=500,
            height=350,
            config={ 
                axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
                legend={
                    labelFont="Times New Roman", titleFont="Times New Roman",
                    labelFontSize=20, symbolSize=150, symbolStrokeWidth=3  
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/Periods$T-Real$num/$algorithm-bounds_Iter_Period$T-Real$num.pdf")
    end
end