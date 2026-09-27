"""Compose the English descriptive report from computed tables and figures.

Requires reportlab and Pillow; no simulator, pandas or experiment dependencies.
The companion Markdown contains the same prose, tables and figure captions.
"""
from pathlib import Path
import csv
import json
import html
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'docs/portugal_rec_native';D=DOCS/'dataset_description'
J=json.loads((D/'statistics.json').read_text());F=J['families'];ORDER=list(F)
W=499.28;INK=colors.HexColor('#203b4b');TEAL=colors.HexColor('#228b85');GOLD=colors.HexColor('#ca8a31');PALE=colors.HexColor('#edf3f4')
for name,file in [('Body','LiberationSans-Regular.ttf'),('BodyBold','LiberationSans-Bold.ttf'),('BodyItalic','LiberationSans-Italic.ttf')]:pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/liberation2/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
S={
 'body':ParagraphStyle('BodyText',fontName='Body',fontSize=10.3,leading=14.5,textColor=INK,spaceAfter=9),
 'small':ParagraphStyle('Small',fontName='Body',fontSize=8.5,leading=11.5,textColor=colors.HexColor('#536b78'),spaceAfter=7),
 'lead':ParagraphStyle('Lead',fontName='Body',fontSize=11.4,leading=16,textColor=INK,spaceAfter=13),
 'title':ParagraphStyle('Title',fontName='BodyBold',fontSize=25,leading=29,textColor=INK,spaceAfter=13),
 'h2':ParagraphStyle('Heading2',fontName='BodyBold',fontSize=12.2,leading=16,textColor=TEAL,spaceBefore=10,spaceAfter=6),
 'eyebrow':ParagraphStyle('Eyebrow',fontName='BodyBold',fontSize=8.5,leading=11,textColor=TEAL,spaceAfter=10),
 'cell':ParagraphStyle('Cell',fontName='Body',fontSize=8.5,leading=11.5,textColor=INK),
 'headcell':ParagraphStyle('HeadCell',fontName='BodyBold',fontSize=8.4,leading=11,textColor=colors.white),
 'caption':ParagraphStyle('Caption',fontName='Body',fontSize=8.5,leading=11.5,textColor=colors.HexColor('#4e6673'),spaceBefore=4,spaceAfter=12),
}
PAGES=[];FIGURES=[]
def n(x,d=0):return f'{x:,.{d}f}'
def p(text,kind='body'):return {'type':'p','text':text,'style':kind}
def h(text):return {'type':'h','text':text}
def table(headers,rows,widths=None):return {'type':'table','headers':headers,'rows':rows,'widths':widths}
def fig(name,caption,maxheight=250):
    ix=len(FIGURES)+1;FIGURES.append((ix,name,caption));return {'type':'figure','name':name,'caption':f'Figure {ix}. '+caption,'maxheight':maxheight}
def page(title,kicker,*items):PAGES.append({'title':title,'kicker':kicker,'items':list(items)})
def read(name):return list(csv.DictReader((D/f'{name}.csv').open()))
def iv(name,k):return F[name]['inventory'][k]
def variants():return {x['variant']:x for x in J['variants']}


def content():
    page('Portugal REC datasets','2023-2024 / DESCRIPTIVE DATA REPORT',
        p('Communities, assets and temporal patterns','lead'),
        p('A two-year collection of synthetic residential and service communities, informed by Portuguese energy, calendar and market sources. Four physical bases support nine scenario configurations at 15-minute resolution.'),
        fig('00_cover','The Premium-100 solar calendar in 2024. Each column is one civil day and each row a local quarter-hour. Colour shows aggregate PV power. Seasonal daylight and day-to-day weather are visible in the inputs.',225),
        table(['4 physical bases','9 configurations','731 days'],[['4, 15, 30 and 100 members','Static and dynamic scenarios','70,176 records per annual-pair series']],[.32,.35,.33]),
        p('Dataset edition: '+J['version']+'<br/>Report prepared: 27 September 2026<br/>Scope: dataset composition and input distributions','small'),
        p('This report describes the published data. All quantities are computed from inventories, schedules, schemas and input time series.'))
    page('Reading the collection','PURPOSE AND READING GUIDE',
        p('The collection brings together households with different daily routines, workplaces and other services. Some members contribute generation or controllable equipment; others contribute consumption alone. The same calendar supports weather, electricity prices, carbon intensity and operating schedules.','lead'),
        p('The report starts with the catalogue and inventories, then follows the data from buildings and energy profiles to vehicles, appliance requests and scenario events. The final pages explain the shared signals, file organisation and source provenance.'),
        table(['Part','Pages','What it describes'],[
            ['Catalogue and construction','3-5','The nine configurations, base inventories and source-to-series construction.'],
            ['Buildings and electricity','6-16','Households, routines, ownership, demand, PV, batteries and electrical connections.'],
            ['Mobility and flexible cycles','17-24','EVs, chargers, visits, requested SOC, home-work links and appliance schedules.'],
            ['Scenario histories','25-29','Membership and asset changes, health, measurement noise, outages and explicit DR requests.'],
            ['Shared signals and documentation','30-33','Prices, carbon, weather, forecast inputs, files, units and sources.']],[.29,.12,.59]),
        h('Three counting conventions'),
        p('<b>Inventory</b> means the complete physical base, as selected by its Normal configuration. <b>Active inventory</b> means assets belonging to members present at a particular time in a dynamic configuration. <b>Requests</b> means declared EV sessions or appliance cycles, rather than measured delivery.'),
        p('Annual demand in this report is the non-shiftable electricity input. Flexible appliance requests are reported separately, as are EV connection and SOC requests. Stationary-battery dispatch and EV charging consumption depend on a subsequent controller and are not included in these exogenous energy totals.'),
        p('The four bases are designed scenarios rather than four independent population samples. Member labels, construction rules and shared external signals recur across bases. The summed catalogue count is not a count of distinct real households.'))
    rows=[]
    v=variants()
    descriptions={
      'core_4':'Small residential base for inspection and setup.', 'core_15':'Residential mix with several daily routines.',
      'core_30_normal':'Mixed residential and service reference.', 'core_30_topology':'Membership and asset lifecycle changes.',
      'core_30_safety':'Selected electrical limits are more restrictive.', 'core_30_robustness':'Health events, observation noise and outages.',
      'core_30_combined':'Topology, electrical stress, health and outages together.',
      'premium_100_normal':'Larger mixed community with a richer asset inventory.', 'premium_100_allin':'Combined events, seasonal noise and explicit demand response.'}
    names=['core_4','core_15','core_30_normal','core_30_topology','core_30_safety','core_30_robustness','core_30_combined','premium_100_normal','premium_100_allin']
    for key in names:
        z=v[key];rows.append([z['label'],f"{z['initial_members']} to {z['final_members']}",descriptions[key]])
    page('The nine configurations','01 / CATALOGUE',
        p('Four physical bases are reused through a compact set of configurations. The 30-member family separates individual scenario dimensions, while the two 100-member configurations provide a static reference and an integrated event history.','lead'),
        table(['Configuration','Active members','Defining input characteristics'],rows,[.28,.17,.55]),
        h('What changes between variants'),
        p('Core-30 Topology and Combined share the membership history. Safety and Combined modify selected electrical settings; the remaining configurations also have explicit electrical limits. Robustness and Combined contain health and outage events. Continuous measurement noise is selected in Core-30 Robustness; Combined uses the health events without that measurement-noise layer.'),
        p('Premium All-in includes membership and asset changes, electrical stress, health, outages, measurement-noise campaigns and DR. Premium Normal has the same full-base equipment catalogue, with the event layers disabled. DR appears only in All-in.'),
        p('The development and reserved confirmation event files are alternative schedules within this catalogue. They do not create additional physical bases or additional dataset names.','small'))
    inventory_rows=[]
    for label,key in [('Members','members'),('Residential / service sites',None),('PV sites','pv_sites'),('PV capacity (kWp)','pv_kwp'),('Stationary batteries','battery_sites'),('Battery energy (kWh)','battery_kwh'),('Charging sites / connectors',None),('Distinct EV identities','evs'),('Controllable appliances','controlled_apps'),('Consumption-only members','pure_consumers')]:
        vals=[]
        for name in ORDER:
            if label=='Residential / service sites':value=f"{n(iv(name,'residential'))} / {n(iv(name,'services'))}"
            elif label=='Charging sites / connectors':value=f"{n(iv(name,'charger_sites'))} / {n(iv(name,'chargers'))}"
            else:value=n(iv(name,key),2 if key in ('pv_kwp','battery_kwh') else 0)
            vals.append(value)
        inventory_rows.append([label]+vals)
    page('Inventory at a glance','02 / THE FOUR PHYSICAL BASES',
        p('The catalogue increases in scale while preserving mixed ownership. Premium contains more storage and charging infrastructure in absolute terms, together with large service installations.','lead'),
        table(['Full static inventory']+ORDER,inventory_rows,[.36,.14,.15,.15,.20]),
        fig('01_overview','Residential/service composition and the scale of PV and stationary storage. Bubble area increases with connector count; exact totals are given in the table.',220),
        p('A charging site can have several connectors. An EV identity can visit more than one site. An appliance count refers to a physical device, whereas each device can issue many cycle requests over the two years. These quantities are therefore kept separate.','small'))
    page('From sources to community data','03 / CONSTRUCTION AND TIME',
        p('The collection combines public external series with a synthetic inventory and operating schedules. The resulting records describe fictional members and equipment on a common chronological axis.','lead'),
        table(['Layer','Construction','Output in the dataset'],[
          ['Members and buildings','Household routines, floor area, available roof area and service archetypes.','Member catalogue, narratives and operating calendars.'],
          ['Electricity demand','E-REDES shape priors, occupancy/routine draws and simplified end-use components.','Non-shiftable load and a component ledger.'],
          ['PV and weather','Lisbon solar resource, temperature and wind; array geometry, shading and inverter ratings.','Quarter-hour weather and per-site absolute PV generation.'],
          ['Mobility and appliances','Declared vehicles and connectors; visits, SOC preferences and integrated cycle profiles.','Connection records, departure requests and start/deadline windows.'],
          ['Market and carbon','OMIE, ERSE/contract components, REN, REE and APA reference information.','Common price and operational carbon-intensity series.'],
          ['Scenario overlays','Schema-defined changes and event files with deterministic timing.','Topology, observation/action/asset events, outages and DR requests.']],[.22,.43,.35]),
        h('A continuous two-year calendar'),
        p('Each base contains 35,040 records for 2023 and 35,136 for leap-year 2024: 70,176 consecutive quarter-hour records. The time index is regular in UTC. Residential and business timing is expressed in Europe/Lisbon, including daylight-saving changes, holidays and school breaks.'),
        p('Local days at daylight-saving transitions contain 92 or 100 quarter-hours. The regular UTC index preserves elapsed time, while civil-time fields preserve the meaning of opening hours and household schedules. 2024 continues the community state reached at the end of 2023; membership is not reset.'),
        p('Public observations, reanalysis and satellite products provide the external context. Individual household loads, visits, cycle requests and event campaigns are constructed scenarios, rather than recordings from the people named in the atlas.','small'))
    page('Households and premises','04 / POPULATION AND BUILDINGS',
        p('The two smaller bases are residential. Core-30 combines 24 households with six services, and Premium combines 80 households with 20 services. The latter include four offices, four retail shops, three clinics, three cafes, three workshops, a sports centre, a grocery store and a care home.','lead'),
        fig('02_households','Household-size distributions and residential floor-area distributions. Boxes show the median and interquartile range; isolated boxplot outliers are hidden for readability.',240),
        table(['Base','Household residents','Home area: min / median / max (m²)'],[[name,n(iv(name,'residents')),f"{n(F[name]['home_area']['min'])} / {n(F[name]['home_area']['median'],1)} / {n(F[name]['home_area']['max'])}"] for name in ORDER],[.23,.25,.52]),
        p('Premium households contain one to five residents, with a median of two. Residential floor area spans 55-179 m². Service premises introduce a separate scale: the main office occupies 650 m², the sports centre 750 m², the grocery store 500 m² and the care home 900 m².'),
        p('The catalogue occupant count describes household residents. A zero in that field for a service site does not describe an empty workplace; service operation is represented through its own activity profile and opening calendar.','small'))
    page('Daily life and opening calendars','05 / ROUTINES',
        p('Residential routines include daytime commuters, hybrid workers, night-shift workers, school-run households, retired residents and people working late. This produces morning returns, intermediate visits and extended daytime presence as well as conventional evening arrivals.','lead'),
        fig('03_routines','Premium residential archetypes and mean opening hours by weekday in 2024. The opening-hour summary includes holidays; it is an average over actual civil days.',260),
        p('Offices close at weekends and on the applicable holidays. Retail, hospitality and other services have their own calendars; weekend operation is therefore not treated as a universal exception. The care home has a continuous operating pattern.'),
        h('Examples from the member atlas'),
        p('<b>Premium 003:</b> a four-person, 131 m² household associated with night-shift work and a home charging connector. <b>Premium 004:</b> a three-person, 122 m² school-run household with PV and a 5.12 kWh stationary battery. <b>Premium 081:</b> a 650 m² office with twelve connectors, PV, a 10.24 kWh battery and controllable appliance cycles.'),
        p('Names and life stories are fictional descriptions of the same catalogue used to construct the data. They give a readable context to schedules and equipment; they are not personal survey records.','small'))
    page('Ownership and mixed participation','06 / ASSET DISTRIBUTION',
        p('Asset ownership overlaps: a member can host PV, storage, charging and flexible cycles in several combinations. Members without any of these assets remain part of the community demand.','lead'),
        fig('04_ownership','Premium ownership by member and asset-ownership shares across all four bases. The horizontal line after Premium member 080 separates households from services. Consumption-only means no PV, stationary battery, EV connector or controllable appliance.',290),
        table(['Base','Consumption only','PV owners','Battery owners','Charging sites'],[[name,f"{n(iv(name,'pure_consumers'))} ({iv(name,'pure_consumers')/iv(name,'members')*100:.1f}%)",f"{iv(name,'pv_sites')/iv(name,'members')*100:.1f}%",f"{iv(name,'battery_sites')/iv(name,'members')*100:.1f}%",f"{iv(name,'charger_sites')/iv(name,'members')*100:.1f}%"] for name in ORDER],[.24,.25,.17,.17,.17]),
        p('Premium has 32 consumption-only members and stationary batteries at 20% of sites. Its richer equipment mix is concentrated in particular households and services; it does not require identical equipment at every site.','small'))
    page('Electricity demand and end uses','07 / ANNUAL INPUT ENERGY',
        p('Non-shiftable demand is supplied directly as interval energy. It combines a base component with fixed appliance cycles and, where present, electric hot-water, heating, cooling and battery-standby components. Controllable appliance cycles are supplied as separate requests.','lead'),
        fig('05_energy','The 2024 non-shiftable demand ledger and each member\'s annual load/PV position in its static base. These annual totals do not estimate self-consumption or grid exports.',240),
        table(['Base','Load 2023 / 2024 (MWh)','PV 2023 / 2024 (MWh)','Flexible cycles 2024 (MWh)'],[[name,f"{iv(name,'load_2023_mwh'):.2f} / {iv(name,'load_2024_mwh'):.2f}",f"{iv(name,'pv_2023_mwh'):.2f} / {iv(name,'pv_2024_mwh'):.2f}",f"{iv(name,'cycles_2024_kwh')/1000:.2f}"] for name in ORDER],[.20,.29,.29,.22]),
        p('The annual columns are sums of input energy. EV charging energy and stationary-battery operation are not added because they require control decisions. The cycle column is requested programme energy; fixed cycles are already inside non-shiftable demand and are not counted again.'),
        p('Heating and cooling are simplified electricity components responsive to the weather. They describe demand variation without introducing a separate building-comfort model.','small'))
    page('Demand through the seasons','08 / MONTHLY PROFILES',
        p('The two years retain the same physical inventory while using their own calendars and external series. Monthly demand reflects activity schedules, vacations and temperature-dependent end uses.','lead'),
        fig('06_seasonal_load','Monthly mean non-shiftable demand per catalogue member. Dashed lines are 2023; solid lines are 2024. Dividing by days in each month avoids treating a longer month as higher daily demand.',285),
        p('Core-4 and Core-15 show the residential shape. Core-30 and Premium include higher-consuming service premises, so their per-member mean is not a household-consumption statistic. The normalisation is useful for viewing changes within a base, rather than assuming that every member represents the same type of building.'),
        p('The 2024 generation procedure retains equipment ratings and daily reference budgets, draws new operating schedules and adjusts heating/cooling components using the temperature record. Leap day, movable holidays, school breaks and civil-time changes are represented explicitly.'),
        p('These curves describe the static bases. Event-conditioned input files additionally apply membership and asset availability in the dynamic configurations.','small'))
    page('Weekdays, weekends and holidays','09 / INTRADAY DEMAND',
        p('Calendar effects are encoded in both the activity rules and the resulting load profiles. Separating residential and service members shows different changes in timing and daytime demand.','lead'),
        fig('07_daily_load','Premium mean non-shiftable power per site in 2024, separated by residential/services and mutually exclusive day classes. Public holidays take precedence over weekend classification.',240),
        p('Residential profiles retain morning and evening activity while the midday shape varies with the day class. Service demand rises with opening hours. Its weekend and holiday aggregates remain nonzero because the service group contains sites such as cafes, retail businesses and the care home.'),
        h('How the average is formed'),
        p('The input interval energy is converted to average power by multiplying by four. Values are grouped by Lisbon civil hour and day class, then averaged across the relevant members. The chart shows the average over many days; it does not imply that every Tuesday or every household follows exactly the same curve.'),
        p('The seasonal and intraday views complement one another: a monthly total describes energy volume, whereas the civil-hour profile describes when that energy is requested.'))
    page('The distribution of demand','10 / MEMBER CONTRIBUTIONS',
        p('Larger service sites account for a different share of energy than individual homes. Ranked energy contributions make this composition visible without conflating a member count with an energy share.','lead'),
        fig('08_concentration','Cumulative 2024 non-shiftable demand, with members ranked from largest to smallest, and the eight largest Premium demand sites. The dotted line represents equal demand per member.',250),
        table(['Base','Largest member','Largest five members','Largest 20% of members'],[[name,f"{F[name]['top_one_load_share']:.1f}%",f"{F[name]['top_five_load_share']:.1f}%",f"{F[name]['top_quintile_load_share']:.1f}%"] for name in ORDER],[.25,.25,.25,.25]),
        p(f"Core-30's largest site represents {F['Core-30']['top_one_load_share']:.1f}% of non-shiftable demand. Premium's largest site represents {F['Premium-100']['top_one_load_share']:.1f}%, while its five largest sites jointly account for {F['Premium-100']['top_five_load_share']:.1f}%. These are descriptive consequences of the catalogue's residential/service mix."),
        p('For Core-4, the largest-five column includes all four members. The largest-20% column uses the ceiling of 20% of the member count. All shares use the same static-base demand denominator.','small'))
    page('PV installations and physical ratings','11 / SOLAR ASSETS',
        p('Arrays are built from integer numbers of 420 Wp modules, subject to available roof area and a separate AC inverter rating. Different tilts, azimuths and shading factors give the sites distinct production profiles under the common weather record.','lead'),
        fig('09_pv_fleet','PV array nameplate capacity by base and Premium DC/AC ratings. Each point is an installation. The dotted line marks equal DC array and AC inverter ratings.',240),
        table(['Base','PV modules','PV sites','Smallest / largest array (kWp)','Total capacity (kWp)'],[[name,n(iv(name,'pv_modules')),n(iv(name,'pv_sites')),f"{F[name]['pv_sizes']['min']:.2f} / {F[name]['pv_sizes']['max']:.2f}",n(iv(name,'pv_kwp'),2)] for name in ORDER],[.20,.15,.13,.30,.22]),
        p('Premium arrays span 2.10-48.30 kWp. Its 36 PV sites comprise 744 modules with 312.48 kWp of total DC capacity. Orientations include east, southeast, south, southwest and west; the available tilt classes are 10, 20, 30 and 35 degrees.'),
        p('The solar_generation field is absolute energy for a quarter-hour interval. It is not a normalised one-kilowatt production factor. Rated array power, inverter power and interval generation are separate quantities.','small'))
    page('Solar seasonality and daylight','12 / PRODUCTION PATTERNS',
        p('Solar production follows the annual light cycle and the weather for each year. The same equipment persists into 2024, allowing differences in the resource and calendar to be visible without changing the array inventory.','lead'),
        fig('10_pv_seasons','Premium monthly specific PV yield and mean January/July 2024 power profiles. Solid intraday curves show PV; dashed curves show non-shiftable load. They are descriptive averages, not a dispatch or sharing calculation.',260),
        table(['Base','2023 yield (kWh/kWp)','2024 yield (kWh/kWp)'],[[name,n(F[name]['pv_yield_2023']),n(F[name]['pv_yield_2024'])] for name in ORDER],[.34,.33,.33]),
        p('The longer summer production window is visible in the daily profiles and in the cover calendar. Different days within the same season still have different solar output; the yearly record is not a repeated clear-sky day.'),
        p('The yield denominator is the full static-base DC capacity. An annual or monthly PV total alone does not determine how much production can be used by the community at the time it occurs.','small'))
    page('Stationary battery inventory','13 / STORAGE ASSETS',
        p('Stationary batteries are discrete assets with energy capacity, power limits, efficiency, initial SOC and self-discharge parameters. Their inventory can be described independently of any charging or discharging policy.','lead'),
        fig('11_batteries','Stationary battery capacities and energy/power pairs. Repeated products may overlap in the scatter plot; the table gives the exact counts.',235),
        table(['Base','Batteries','Total energy (kWh)','Total rated power (kW)','Capacity classes (kWh: count)'],[[name,n(iv(name,'battery_sites')),n(iv(name,'battery_kwh'),2),n(iv(name,'battery_kw'),2),'; '.join(f'{float(k):g}: {v}' for k,v in F[name]['batteries_by_size'].items())] for name in ORDER],[.19,.13,.20,.21,.27]),
        p('Premium includes nine 5.12 kWh batteries, nine 10.24 kWh batteries and two 40.96 kWh batteries. The total is 220.16 kWh and 108.40 kW. The two larger products are located at the sports centre and care home.'),
        p('The catalogue uses a 90% depth-of-discharge setting and a 0.94 efficiency parameter, with explicit power curves and self-discharge. Physical battery cycle counts, annual throughput and lifetime use are outcomes of later operation and are not properties inferred in this report.','small'))
    page('Electrical connections and phases','14 / SITE LIMITS',
        p('Every member has an electrical-service definition. It records the connection type, total import/export limits and the limits assigned to each phase. Equipment ratings and connection limits are therefore separate parts of the data.','lead'),
        fig('12_connections','Single/three-phase site proportions and the selected Core-30 import-limit changes in Safety. Points below the dotted line have a reduced limit; overlapping points correspond to repeated connection levels.',240),
        table(['Base','Single phase','Three phase'],[[name,n(F[name]['single_phase_sites']),n(F[name]['three_phase_sites'])] for name in ORDER],[.4,.3,.3]),
        p('Single-phase sites assign the connection capacity to one phase and use zero capacity on the unused phases. Three-phase sites specify capacity across L1, L2 and L3. The supplied active-power convention uses a power factor of one; the connection catalogue records that assumption.'),
        p('Core-30 Safety changes selected limits while retaining the base equipment and demand inputs. Combined uses the same electrical-stress layer alongside its lifecycle and health history. Normal configurations also have explicit physical limits; the Safety label identifies a changed scenario, not the first introduction of limits.'),
        p('Per-phase quantities describe the declared connection and asset allocation. This is an aggregate active-power dataset, rather than a feeder voltage or protection-coordination study.','small'))
    page('Electric vehicles and battery sizes','15 / THE VEHICLE CATALOGUE',
        p('Vehicles have persistent identities in the catalogue, with usable battery capacity, accepted AC power and SOC preferences. Resident vehicles are associated with a home member; additional vehicle pools represent visitors to service sites.','lead'),
        fig('13_ev_fleet','EV usable-capacity distributions and the split between resident vehicles and external visitors. Cumulative curves count vehicle identities, not charging sessions.',240),
        table(['Base','Residents / visitors','EV capacity range (kWh)','Home-work linked identities'],[[name,f"{n(iv(name,'resident_evs'))} / {n(iv(name,'visitor_evs'))}",f"{min(map(float,F[name]['ev_capacities'])):g}-{max(map(float,F[name]['ev_capacities'])):g}",n(F[name]['home_work_links'])] for name in ORDER],[.22,.27,.27,.24]),
        p('Premium contains 85 vehicle identities: 37 resident vehicles and 48 visitors. Its battery classes extend from 40 to 105 kWh, including 77, 82 and 95 kWh products. Maximum accepted AC power is declared separately from battery energy capacity.'),
        p('A vehicle with a large battery can be connected to a modest-power home charger. The vehicle limit and the connector rating both form part of the input specification; neither guarantees that a particular departure request will be delivered.','small'))
    page('Where charging infrastructure sits','16 / CONNECTORS AND SITES',
        p('Charging is distributed between homes and services. Connector counts vary by site, including households with two connectors and a larger office charging location. The connector catalogue preserves individual identities and phase connections.','lead'),
        fig('14_chargers','The ten Premium sites with the most connectors, and connector counts by rated charging power and phase assignment. all_phases denotes a three-phase connection.',265),
        p('Premium has 65 connectors across 41 sites. Office 081 has twelve; Office 086 and the sports centre each have four. Other sites include two-connector homes and smaller service charging locations.'),
        table(['Premium connector rating (kW)','3.68','7.36','11.04','22.08'],[['Connector count']+[str(F['Premium-100']['chargers_by_power'].get(str(x),0)) for x in [3.68,7.36,11.04,22.08]]],[.36,.16,.16,.16,.16]),
        p('The published connector catalogue sets maximum discharging power to zero. Vehicle-to-grid export is not part of these connector specifications. Charging availability is represented separately through vehicle connection records.','small'))
    page('Arrivals and departures','17 / SESSION TIMING',
        p('The connection schedules contain morning, daytime and evening movements. Residential and service locations have different timing distributions, and the residential group includes routines that overlap with conventional office hours.','lead'),
        fig('15_ev_times','Premium 2024 arrival and departure hours for complete sessions in the Normal input files. Histograms use one-hour bins in Lisbon civil time. Sessions crossing the start or end of the two-year record are excluded here.',255),
        table(['Base','Sessions starting in 2023','Sessions starting in 2024'],[[name,n(F[name]['sessions_by_year']['2023']),n(F[name]['sessions_by_year']['2024'])] for name in ORDER],[.34,.33,.33]),
        p('Session counts refer to visits reconstructed from each Normal configuration\'s connector records. A visit spanning New Year remains one visit and is attributed to its start year. This preserves the distinction between a new arrival and a continuation of existing parking.'),
        p('A connection interval gives the period during which charging may be available. It does not indicate that a vehicle charges continuously for the full interval.','small'))
    page('Requested SOC and connection duration','18 / SERVICE REQUESTS',
        p('Departure targets vary by visit. They combine the user preference and preparation for forthcoming travel, producing a spread of requested SOC values rather than one universal target. All declared targets remain within the accepted 50-100% range.','lead'),
        fig('16_soc_dwell','Premium 2024 target-SOC bands and cumulative connection-duration distributions for complete sessions. The final SOC band is exactly 100%. The duration axis displays the first 36 hours; residential parking has a longer tail.',245),
        table(['Base','Median target SOC','Sessions requesting 100%','Median / 90th-percentile dwell (h)'],[[name,f"{F[name]['soc_target']['median']:.1f}%",f"{F[name]['target_100_pct']:.2f}%",f"{F[name]['session_duration']['median']:.2f} / {F[name]['session_duration']['p90']:.2f}"] for name in ORDER],[.20,.22,.25,.33]),
        p('The table uses complete sessions across both years. A long parking interval can include time when no additional charging is needed. Likewise, a high requested SOC is a service request rather than an observation of energy delivered.'),
        p('Arrival SOC is supplied in the input records. Requested energy gaps can be described from capacity and the difference between arrival and target SOC, but an annual charging-energy total requires the control trajectory.','small'))
    page('Connection availability across the week','19 / WHEN FLEXIBILITY IS PRESENT',
        p('A weekly view complements the arrival and departure histograms. It shows how many connectors are occupied at the same time, averaged over the year, and how that pattern differs between homes and services.','lead'),
        fig('17_connection_calendar','Mean fraction of Premium connectors with a vehicle connected, by local hour and weekday in 2024. Residential and service connectors use separate denominators. These are scheduled connections, not charging utilisation.',270),
        p('Residential connectors are more often occupied overnight, while service connectors concentrate their occupancy in operating hours. Some home connectors remain occupied during the day because the household routines include working from home, school-run returns, retirement and shift work.'),
        p('Service availability reflects a mix of offices, retail and other sites. Weekend connection rates are therefore shaped by the composition and opening calendars, not by an assumption that all services behave like an office.'),
        h('Interpreting the colour scale'),
        p('A value of 50% means that, in the selected hour/weekday group, half of the available connector records are in the connected state on average. It does not mean half of installed charging power is being used.'))
    page('Home-work links and recurring visitors','20 / MOBILITY STRUCTURE',
        p('A subset of resident vehicles has a declared work destination inside the community. This links home and workplace availability for the same vehicle identity, alongside separate recurring visitor pools.','lead'),
        fig('18_mobility_links','The four resident vehicle identities with a home-work link in Premium. Each edge joins a home member to the designated work member; repeated daily journeys are not drawn separately.',230),
        p('Premium links homes 001, 026, 038 and 050 to Office 081. Core-30 includes one home-work link. The smaller residential bases do not contain a workplace member.'),
        p('External visitors also have vehicle identities and equipment attributes. They can appear repeatedly at a service site without becoming a resident member of the community. The distinction between visitors and members keeps the mobility population separate from the REC membership count.'),
        p('The visit data support different lengths of service stop, including split working-day visits. Vehicle identity and the connection calendar describe where a car is available; arrival SOC is prescribed for each session in the original simulator contract. The report therefore does not treat the mobility catalogue as a measured longitudinal battery-energy balance.','small'))
    page('Appliance programmes and cycles','21 / FLEXIBLE APPLIANCES',
        p('The appliance catalogue contains washing machines and dishwashers. Some follow fixed schedules already incorporated into non-shiftable demand; others expose their cycle-start decision through the controllable appliance interface.','lead'),
        fig('19_appliances','Illustrative Premium washing-machine and dishwasher programmes, and the energy/duration distribution of all controllable programme profiles. Powers are interval averages obtained from the stored quarter-hour cycle energy.',255),
        table(['Base','Controlled devices','Fixed devices','Requested cycles, 2023 / 2024'],[[name,n(iv(name,'controlled_apps')),n(iv(name,'fixed_apps')),f"{n(F[name]['cycles_by_year']['2023'])} / {n(F[name]['cycles_by_year']['2024'])}"] for name in ORDER],[.21,.23,.23,.33]),
        p('Cycle profiles preserve the sequence of higher-power heating stages and lower-power motor or waiting stages at 15-minute resolution. Different programmes have different durations and energy requirements. These are integrated synthetic waveforms rather than recordings of individual appliance models.'),
        p('The published appliance classes are washing machines and dishwashers; a separate tumble-dryer dependency chain is not part of this catalogue.','small'))
    page('Windows for starting a cycle','22 / TIME FLEXIBILITY',
        p('Each controllable request identifies a programme, earliest start, latest start and completion deadline. This describes when the service may occur while retaining the programme\'s ordered energy profile.','lead'),
        fig('20_flexibility','Allowed start-time slack in 2024: latest start minus earliest start. Left: distribution by base. Right: Premium weekday/weekend/holiday requests. Boxplots show medians, quartiles and non-outlier whiskers.',250),
        table(['Base','Median start slack (h)','10th-90th percentile (h)','Median cycle duration (h)'],[[name,f"{F[name]['cycle_slack']['median']:.2f}",f"{F[name]['cycle_slack']['p10']:.2f}-{F[name]['cycle_slack']['p90']:.2f}",f"{F[name]['cycle_duration']['median']:.2f}"] for name in ORDER],[.22,.26,.27,.25]),
        p('The table summarises both years; the chart isolates 2024. The range includes narrow or fixed-start requests as well as windows extending up to eight hours. Calendar-dependent windows and different household preferences provide variation between requests.'),
        p('Start slack is not interchangeable with cycle duration or deadline slack. A long cycle can have a narrow start window, and a broad start window does not change the energy required by its selected programme.','small'))
    active=[]
    for key in ['core_30_topology','premium_100_allin']:
        x=v[key];active.append([x['label'],f"{x['initial_members']} to {x['final_members']}",f"{x['initial_assets']['pv']} to {x['final_assets']['pv']}",f"{x['initial_assets']['batteries']} to {x['final_assets']['batteries']}",f"{x['initial_assets']['chargers']} to {x['final_assets']['chargers']}",f"{x['initial_assets']['appliances']} to {x['final_assets']['appliances']}"])
    page('A community that changes over time','23 / MEMBERSHIP AND ASSET HISTORIES',
        p('Dynamic configurations represent first admissions, permanent departures, equipment commissioning, replacement and retirement. These events change the inventory present in the scenario, rather than repeatedly switching the same departed member in and out.','lead'),
        fig('21_topology','Active member trajectories and the Premium All-in asset inventory. Asset counts include only active members. The unchanged 2024 segment carries forward the state reached in 2023.',255),
        table(['Configuration','Members','PV sites','Batteries','Connectors','Flexible devices'],active,[.26,.14,.14,.14,.16,.16]),
        p('Core-30 Topology has four admissions and three departures, moving from 26 to 27 active members. Combined shares this history. Premium All-in has twenty admissions and twelve departures, moving from 80 to 88.'),
        p('The topology events occur in 2023. They are not replayed in 2024, and departed members do not re-enter. Health, outage, noise and DR campaigns have their own two-year schedules, separate from this permanent membership history.','small'))
    page('Events behind the member stories','24 / LIFECYCLE EXAMPLES',
        p('The atlas and the lifecycle catalogues are generated from the executable schema events. A member\'s description therefore distinguishes the static configuration from a variant in which its participation or equipment changes.','lead'),
        fig('22_member_history','Two histories for member 001, with Lisbon civil timestamps. The matching member label belongs to a different base in each row; the dates are not assumed to be identical.',245),
        p('In Core-30 Topology and Combined, the first stationary battery is removed on 14 May and its replacement is installed on 16 May. Member 001 leaves on 14 November at 22:00. In Premium All-in, the battery replacement occurs on 16-17 May and the member leaves on 14 August at 13:00.'),
        p('A replacement is a new physical asset identity. A permanent departure removes the member and its associated activity from the scenario. Retired appliances no longer generate future service requests; outstanding or interrupted cycles are cancelled according to the supplied lifecycle treatment.'),
        p('Static variants retain their catalogue membership. The same member\'s narrative therefore names the configuration in which each transition applies, instead of presenting one date as universal across the family.'),
        p('The event catalogue excludes internal template-registration operations from the real-world narrative. Templates used to introduce a later asset are not described as physical retirements.','small'))
    page('Health events and measurement noise','25 / INFORMATION AND EQUIPMENT EVENTS',
        p('Health events and measurement noise are stored as separate input layers. Health includes missing observations, biased price forecasts, delayed or dropped commands and unavailable assets. Measurement noise perturbs selected observation channels using the configured distribution and seed.','lead'),
        fig('23_health_calendar','Development schedules across both years. Thin events are marked with points for visibility; marker width is not event duration. Core-30 Robustness has continuous measurement noise; Premium All-in has seasonal campaigns.',270),
        p('The measurement settings are 0.005 kWh per interval for non-shiftable load, 0.020 kW for PV power and 0.005 SOC fraction for selected EV/battery channels. These correspond to 20 W and 0.5 percentage points for the respective channels. Domain clipping keeps the declared observation values within the specified ranges.'),
        p('Premium uses four 14-day development campaigns per year: 56 days per year and 112 days across the collection. Core-30 Combined includes health events without the continuous measurement-noise layer.'),
        p('A logical event can appear on several channels. Event-row counts are consequently different from distinct campaign counts or elapsed exposure. Reserved confirmation files provide a separate event schedule.','small'))
    o=read('outages');outrows=[]
    for key in ['core_30_robustness','core_30_combined','premium_100_allin']:
        rr=[x for x in o if x['variant']==key];z=v[key];outrows.append([z['label'],str(z['outage_sites']),str(len(rr)),n(sum(float(x['duration_h']) for x in rr))])
    page('Scheduled grid interruptions','26 / OUTAGE INPUTS',
        p('Grid availability is supplied as an interval-level input at the affected sites. Outage episodes cover different calendar periods and local times, and are distinct from an equipment-availability fault.','lead'),
        fig('24_outages','Premium All-in outage timing and cumulative scheduled site-hours in the three configurations containing outages. A site-hour counts one affected site for one hour; simultaneous sites are counted separately.',240),
        table(['Configuration','Distinct affected sites','Site-level episodes','Scheduled site-hours'],outrows,[.34,.22,.23,.21]),
        p('The global union of outage times is 32 hours across the two years in each of these configurations. The site-hour totals differ because several members can be affected simultaneously and membership changes can end a site\'s subsequent participation.'),
        p('Stationary batteries are included in the backup convention used by the supplied scenarios. Backup availability is part of the scenario definition; the amount of demand actually served during an interruption depends on the battery state and subsequent operation. No served-energy or recovery result is inferred here.','small'))
    drrows=[]
    for r in J['dr'][:4]:
        drrows.append([r['request_id'].removeprefix('development_'),r['direction'],n(r['target_power_kw']),(r['end_time_step']-r['start_time_step']+1)/4,r['stratum'].replace('_',' ')])
    page('Explicit demand-response requests','27 / PREMIUM ALL-IN',
        p('Demand response is represented by explicit requests, independently of the electricity-price sign. Each request provides a direction, active interval, target power and commercial parameters. DR is selected only in Premium All-in.','lead'),
        fig('25_dr','The operational-candidate requests in 2023 and 2024. Positive values request increased demand; negative values request a reduction. The separate 1,500 kW stress request is omitted from this axis to preserve the operational scale.',210),
        table(['Pattern in each year','Direction','Target (kW)','Duration (h)','Declared stratum'],drrows,[.25,.16,.17,.17,.25]),
        p('Each year includes two one-hour reduction requests at 80 kW and one one-hour increase request at 40 kW. A separate four-hour reduction request at 1,500 kW is explicitly labelled impossible_stress. This label is retained as part of the request design rather than pooled with the operational candidates.'),
        p('The files contain an activation price of EUR 0.12/kWh, a shortfall penalty of EUR 0.06/kWh and a 5% power tolerance. These are request parameters. The chart and table describe the requested service; they contain no estimate of delivered response, baseline quality, settlement or rebound.','small'))
    page('Electricity prices and carbon intensity','28 / SHARED EXTERNAL SIGNALS',
        p('All members use the same variable grid-purchase price series. It combines OMIE with the reference contract and regulated access components for the relevant year. Carbon intensity is a separate time-aligned operational estimate based on the national production mix and an imports proxy.','lead'),
        fig('26_price_carbon','Daily mean price and carbon intensity over both years. Shading gives the within-day 10th-90th percentile. Price units are cents/kWh; carbon units are gCO2e/kWh.',280),
        table(['Year','Mean price (EUR/kWh)','Price range (EUR/kWh)','Mean carbon (gCO2e/kWh)'],[[str(r['year']),f"{r['price_mean']:.4f}",f"{r['price_min']:.4f} to {r['price_max']:.4f}",f"{r['carbon_mean_g']:.1f}"] for r in J['market']],[.13,.28,.33,.26]),
        p('Negative values in the reference 2023 variable price are preserved. The series is before VAT and excludes fixed charges. The dataset schemas leave the community market disabled and set grid-export remuneration to zero; the local-market ratio belongs to the later experiment configuration.'),
        p('Carbon is a derived average operational intensity, not a marginal-emissions or lifecycle accounting series. The component files retain the production/import calculation and its provenance.','small'))
    forecast=json.loads((ROOT/'results/portugal_rec_native/forecasts_and_stories.json').read_text())['forecast_metrics']
    forecastrows=[]
    for key,label,unit in [('outdoor_dry_bulb_temperature','Air temperature','°C'),('outdoor_relative_humidity','Relative humidity','percentage points'),('direct_solar_irradiance','Direct irradiance, daytime','W/m²'),('electricity_pricing','Grid purchase price','cents/kWh')]:
        q=[x for x in forecast if x['signal']==key and x['year']==2024 and 'month' not in x];forecastrows.append([label]+[f"{x['mae']:.2f}" for x in q]+[unit])
    page('Weather variation and forecast inputs','29 / THE EXTERNAL CONTEXT',
        p('Both years contain their own weather and market series. Solar, temperature and calendar changes act on the same equipment inventory. Forecast columns expose simple causal point estimates alongside the realised inputs.','lead'),
        fig('27_weather','Daily mean air temperature in both years and Premium daily PV energy in 2024. The PV record contains day-to-day fluctuations within the seasonal envelope.',210),
        table(['2024 forecast input','1 h MAE','6 h MAE','24 h MAE','Unit'],forecastrows,[.34,.13,.13,.13,.27]),
        p('Weather forecasts use daily persistence with an astronomical night mask for solar channels. Published day-ahead OMIE is used when available, with prior-day persistence otherwise. Daily persistence gives almost the same error for a common destination across horizons; it is a reference forecast method rather than a forecast ensemble.'),
        p('The forecast table compares stored inputs with realised future samples, excluding the first and last day. Solar error uses daytime targets. One external support day completes final price-forecast destinations without adding a third simulated year. Humidity is a temperature-derived proxy in 2023 and ERA5 in 2024; that source difference is retained in the documentation.','small'))
    page('Files, units and reproducibility','30 / USING THE DATA DESCRIPTION',
        p('The report is accompanied by computed tables and reusable figures. Their input identity is tied to the published dataset checksums, so the description can be regenerated when the catalogue changes.','lead'),
        table(['Location inside each dataset','Content'],[
          ['schema.json; evaluation/*.json','Default configuration and the current executable variants.'],
          ['catalogs/','Members, equipment, electrical services, annual totals and the derived lifecycle plan.'],
          ['timeseries/; end_uses/','Per-member physical series and the non-shiftable component ledger.'],
          ['shared/','Calendar, weather, price and carbon input series.'],
          ['chargers/; deferrables/; mobility/','Connection and cycle files, with supporting mobility records.'],
          ['evaluation/native_inputs/','Files actually selected by each original-engine configuration.'],
          ['evaluation/events/','Development and reserved confirmation event files.'],
          ['MEMBER_STORIES.md; sources/','Generated member narratives and clearly identified historical source material.']],[.40,.60]),
        h('Units and denominators'),
        p('Physical load, PV and cycle profiles use kWh per 15-minute interval; average kW equals interval kWh multiplied by four. Asset capacities use kWh, equipment power uses kW, PV nameplate uses kWp, SOC is a fraction in the files, prices use EUR/kWh and carbon uses kgCO2e/kWh.'),
        p('Counts in this report are recomputed from current records. Historical annual_* catalogue fields are not substituted for two-year totals. Session and cycle distributions use the files selected by Normal schemas; dynamic event counts use their own schemas.'),
        p('Supporting outputs: dataset_description/statistics.json, CSV tables and figures/descriptive/ in this documentation folder. The collection retains 2,828 static two-year input checks and 213 forecast/story checks. Those checks describe construction integrity; the present report contains no controller-performance measurements.','small'))
    page('Sources and construction references','31 / PROVENANCE',
        p('The source archives and construction scripts accompany the published collection. The links below identify the principal providers and documents; the local source manifests retain the exact downloaded files, request parameters and hashes.','lead'),
        p('<b>[S1] Dataset release.</b> '+J['version']+'. Four physical bases and nine schemas. Manifest: results/portugal_rec_native/manifest.json. The original simulator revision is pinned in the manifest; only dataset construction and descriptive tools are used here.'),
        p('<b>[S2] E-REDES load-shape priors.</b> <link href="https://www.e-redes.pt/sites/eredes/files/2023-01/E-REDES_Perfil_Consumo_2023.xlsx" color="#228b85">Consumption profiles 2023</link> and <link href="https://www.e-redes.pt/sites/eredes/files/2023-12/E-REDES_Perfil_Consumo_2024.xlsx" color="#228b85">consumption profiles 2024</link>. Published chronological intervals, aligned to the common UTC axis.'),
        p('<b>[S3] Solar resource and weather.</b> <link href="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en" color="#228b85">European Commission PVGIS</link>, SARAH-3 and the archived PVGIS 5/6 queries for Lisbon, 38.72° N, 9.14° W. <link href="https://open-meteo.com/en/docs/historical-weather-api" color="#228b85">Open-Meteo historical ERA5 interface</link> for 2024 temperature, humidity, wind and documented solar gap fills. Hourly inputs are interpolated to 15 minutes.'),
        p('<b>[S4] Wholesale prices.</b> <link href="https://www.omie.es/pt/market-results/daily/daily-market/day-ahead-price" color="#228b85">OMIE day-ahead market</link>, Portuguese price zone. Market periods are interpreted in Europe/Madrid before UTC alignment. Archived adjustment data supply the 2023 MIBEL component.'),
        p('<b>[S5] Tariff components.</b> <link href="https://www.erse.pt/atividade/regulacao/tarifas-e-precos-eletricidade" color="#228b85">ERSE electricity tariffs</link>, 2023 and 2024 schedules and their within-year revisions. The archived Coopérnico reference formula supplies the benchmark commercial components; <link href="https://www.coopernico.org/artigo/366" color="#228b85">the 2024 TSE explanation</link> documents that year\'s charge. December 2024 ERSE announcements support January forecast destinations.'),
        p('<b>[S6] Electricity carbon estimate.</b> <link href="https://datahub.ren.pt/en/api-instructions" color="#228b85">REN Data Hub</link> production and exchange series; <link href="https://www.ree.es/en/datos/apidata" color="#228b85">REE API</link> for the Spanish imports proxy; <link href="https://apambiente.pt/clima/fator-de-emissao-de-gases-de-efeito-de-estufa-para-eletricidade-produzida-em-portugal" color="#228b85">APA electricity-emission reference</link> for calibration. The fixed 2023 thermal coefficient is retained for the 2024 derived estimate.'),
        p('<b>[S7] Civil and school calendars.</b> Portuguese national and Lisbon municipal holidays; the applicable school calendars, including Despachos 8356/2022 and 8368/2024. The synthetic school schedule chooses a declared start date within the permitted interval.'),
        p('<b>Companion documentation.</b> Fontes_e_metodo.md, Extensao_2024.md, Tarifas_e_mercado.md, Reproduzir.md, Previsoes_e_historias.md and Atlas_Membros_Premium.md provide the detailed construction record. Figures and statistical summaries in this report are computed from the current dataset edition, not reused from previous experiment reports.','small'))


class Report(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='Title':
            text=flowable.getPlainText();key='page-'+str(self.page);self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,0,False)


def footer(c,doc):
    c.saveState();width,height=doc.pagesize
    if doc.page>1:
        c.setFillColor(TEAL);c.rect(48,height-33,28,2,fill=1,stroke=0)
        c.setFillColor(INK);c.setFont('BodyBold',8);c.drawString(84,height-35,'PORTUGAL REC  |  DATASET DESCRIPTION')
    c.setStrokeColor(colors.HexColor('#d9e3e6'));c.line(48,39,width-48,39)
    c.setFillColor(colors.HexColor('#536b78'));c.setFont('Body',7.6);c.drawString(48,26,'2023-2024  /  v3.1  /  Descriptive input statistics');c.drawRightString(width-48,26,f'{doc.page:02d}')
    c.restoreState()


def make_table(node):
    rows=[[Paragraph(escape(str(x)),S['headcell']) for x in node['headers']]]+[[Paragraph(escape(str(x)),S['cell']) for x in row] for row in node['rows']]
    widths=node['widths'] or [1/len(node['headers'])]*len(node['headers'])
    out=Table(rows,colWidths=[W*x for x in widths],hAlign='LEFT',repeatRows=1)
    out.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#d6e0e4'))]))
    return out


def markdown():
    import re
    def plain(s):
        s=s.replace('<br/>','\n').replace('<b>','**').replace('</b>','**')
        s=re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',r'[\2](\1)',s)
        return html.unescape(s)
    out=[]
    for ix,page in enumerate(PAGES,1):
        out.extend([f"# {page['title']}",f"*{page['kicker']} | PDF page {ix}*",''])
        for node in page['items']:
            if node['type']=='p':out.extend([plain(node['text']),''])
            elif node['type']=='h':out.extend(['## '+node['text'],''])
            elif node['type']=='figure':out.extend([f"![{node['caption']}](figures/descriptive/{node['name']}.png)",node['caption'],''])
            elif node['type']=='table':
                out.append('| '+' | '.join(node['headers'])+' |');out.append('| '+' | '.join('---' for _ in node['headers'])+' |')
                out.extend('| '+' | '.join(str(x) for x in row)+' |' for row in node['rows']);out.append('')
    (DOCS/'Portugal_REC_Dataset_Report.md').write_text('\n'.join(out))


def main():
    content();story=[]
    for index,page in enumerate(PAGES):
        if index:story.append(PageBreak())
        story.append(Paragraph(page['kicker'],S['eyebrow']))
        title_style=S['title']
        if index==0:
            title_style=ParagraphStyle('Title',parent=S['title'],fontSize=36,leading=40,spaceAfter=15)
            story.append(Spacer(1,28))
        story.append(Paragraph(page['title'],title_style))
        for node in page['items']:
            kind=node['type']
            if kind=='p':story.append(Paragraph(node['text'],S[node['style']]))
            elif kind=='h':story.append(Paragraph(node['text'],S['h2']))
            elif kind=='table':story.extend([make_table(node),Spacer(1,12)])
            elif kind=='figure':
                path=DOCS/'figures/descriptive'/f"{node['name']}.png";im=PILImage.open(path);ratio=im.height/im.width;width=W;height=width*ratio
                if height>node['maxheight']:height=node['maxheight'];width=height/ratio
                story.append(KeepTogether([Image(str(path),width=width,height=height,hAlign='CENTER'),Paragraph(escape(node['caption']),S['caption'])]))
    out=DOCS/'Portugal_REC_Dataset_Report.pdf'
    doc=Report(str(out),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=57,bottomMargin=53,title='Portugal REC Datasets 2023-2024: A Descriptive Data Report',author='Portugal REC dataset documentation',subject=J['version'],pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    markdown();(D/'report_structure.json').write_text(json.dumps({'pages':[{'page':i+1,'title':p['title']} for i,p in enumerate(PAGES)],'figures':[{'number':i,'file':f,'caption':c} for i,f,c in FIGURES]},indent=2)+'\n')
    print(json.dumps({'pdf':str(out),'planned_pages':len(PAGES),'figures':len(FIGURES)},indent=2))


if __name__=='__main__':main()
