# Trendfollowing-vwap
Vi vill utnyttja expansioner över vwap för att fånga trender. VWAP är bra för mean reversion strategier då det ger ett average price, men i ett trendföljande system måste vi ha fler parametrar för att kunna definiera expansion. Vårat mean reversion system som använder 
vwap ska handla på EURUSD, GBPUSD, USDCHF och USDCAD under asia session 22:00 - 03:00. Forex tenderar att vara mer mean reverting än till exempel aktie index. Men det finns vissa valutapar som är undantag, i våran mean reversion testing presterade USDJPY sämra än de 
andra paren. Troligtvis eftersom JPY par ofta expanderar mer än de andra paren. Därför kommer vi bygga detta trendföljande system runt JPY par, vilket förhoppningsvis ska ge bra diversifiering i portföljen och komplettera mean reversion systemet. Först och främst innan
vi börjar med projektet måste vi ha en grundlig trade idé som har edge. Vi nämnde tidigare att vi måste definiera expansion genom en tilläggande parameter som inte är vwap. Våran första grund idé som visat edge i in sample har sådanhär logik: för longs ska 1 timmes
bar stänga ovanför gårdagens high, detta är vår expansions definiering. När denna breakout skett ska föregående 5 minuters bar stänga ovanför vwap för att nuvarande 5 minuters bar ska kunna göra en pullback till vwap där barens low tradear under vwap men stänger ovanför. När baren stänger ovanför bildas en long signal där entry skär på nästa bar open. Exits går igenom när en 5 minuters bar stänger under vwap. Logiken är exakt omvänd för shorts.
Innan vi går in i själva projektet testade vi denna grund idén på USDJPY utan sessionsfilter mellan åren 2012 - 2019 som in sample data:
--- STATS ---
Market: USDJPY
Trades: 1370
Total PnL (points): 31.6260
Gross Profit: 72.5984
Gross Loss: -40.9724
Profit Factor: 1.7719
Winrate: 0.3182
Avg Win: 0.1665
Avg Loss: -0.0439
Expectancy (avg/trade): 0.0231
Max Drawdown (points): 1.7438
Max Losing Streak (trades): 16
Sharpe (trade-level): 4.6406
--- PER-ÅR STATS ---
USDJPY - 2012:
Trades: 195
Total PnL (points): 1.9085
Gross Profit: 6.7082
Gross Loss: -4.7997
Profit Factor: 1.3976
Winrate: 0.2974
Avg Win: 0.1157
Avg Loss: -0.0350
Expectancy (avg/trade): 0.0098
Max Drawdown (points): 0.7695
Max Losing Streak (trades): 9
Sharpe (trade-level): 1.1286
USDJPY - 2013:
Trades: 194
Total PnL (points): 9.1146
Gross Profit: 15.6790
Gross Loss: -6.5644
Profit Factor: 2.3885
Winrate: 0.3608
Avg Win: 0.2240
Avg Loss: -0.0529
Expectancy (avg/trade): 0.0470
Max Drawdown (points): 0.8201
Max Losing Streak (trades): 10
Sharpe (trade-level): 2.6868
USDJPY - 2014:
Trades: 173
Total PnL (points): 4.8977
Gross Profit: 8.5201
Gross Loss: -3.6224
Profit Factor: 2.3521
Winrate: 0.3410
Avg Win: 0.1444
Avg Loss: -0.0318
Expectancy (avg/trade): 0.0283
Max Drawdown (points): 0.5571
Max Losing Streak (trades): 12
Sharpe (trade-level): 2.2008
USDJPY - 2015:
Trades: 162
Total PnL (points): 1.6178
Gross Profit: 6.6602
Gross Loss: -5.0424
Profit Factor: 1.3208
Winrate: 0.2963
Avg Win: 0.1388
Avg Loss: -0.0442
Expectancy (avg/trade): 0.0100
Max Drawdown (points): 0.7096
Max Losing Streak (trades): 14
Sharpe (trade-level): 0.9603
USDJPY - 2016:
Trades: 159
Total PnL (points): 4.6021
Gross Profit: 11.8295
Gross Loss: -7.2274
Profit Factor: 1.6368
Winrate: 0.2830
Avg Win: 0.2629
Avg Loss: -0.0634
Expectancy (avg/trade): 0.0289
Max Drawdown (points): 1.7438
Max Losing Streak (trades): 14
Sharpe (trade-level): 1.3102
USDJPY - 2017:
Trades: 176
Total PnL (points): 4.5704
Gross Profit: 10.4204
Gross Loss: -5.8500
Profit Factor: 1.7813
Winrate: 0.3182
Avg Win: 0.1861
Avg Loss: -0.0487
Expectancy (avg/trade): 0.0260
Max Drawdown (points): 1.1615
Max Losing Streak (trades): 9
Sharpe (trade-level): 1.7749
USDJPY - 2018:
Trades: 167
Total PnL (points): 2.6833
Gross Profit: 7.4053
Gross Loss: -4.7220
Profit Factor: 1.5683
Winrate: 0.3413
Avg Win: 0.1299
Avg Loss: -0.0429
Expectancy (avg/trade): 0.0161
Max Drawdown (points): 0.7365
Max Losing Streak (trades): 7
Sharpe (trade-level): 1.5764
USDJPY - 2019:
Trades: 144
Total PnL (points): 2.2316
Gross Profit: 5.3757
Gross Loss: -3.1441
Profit Factor: 1.7098
Winrate: 0.2986
Avg Win: 0.1250
Avg Loss: -0.0311
Expectancy (avg/trade): 0.0155
Max Drawdown (points): 0.5751
Max Losing Streak (trades): 12
Sharpe (trade-level): 1.4243
Det ser väldigt lovande ut och vi kommer börja testa och optimera denna strategin ytterligare med hårdare kostnader, optimeringsfilter, out of sample tests och mer i projektloggen.
