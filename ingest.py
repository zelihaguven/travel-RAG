import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings import ChromaDefaultEmbeddings

SAMPLE_REVIEWS = [
    # ── PARIS ──────────────────────────────────────────────────────────────────
    {"city": "Paris", "type": "hotel", "name": "Hotel Le Marais Boutique", "budget": "mid",
     "review": "Perfect location in Le Marais district. Clean rooms, friendly staff. €120/night including breakfast. Walking distance to Centre Pompidou and Place des Vosges. Highly recommended for couples."},
    {"city": "Paris", "type": "hotel", "name": "Ibis Paris Gare de Lyon", "budget": "budget",
     "review": "Great budget option near Gare de Lyon. €70/night, simple but clean. Metro access is excellent. Good for solo travelers on a tight budget."},
    {"city": "Paris", "type": "hotel", "name": "Hotel de Crillon", "budget": "luxury",
     "review": "Magnificent 5-star palace hotel on Place de la Concorde. €800/night. Impeccable service, stunning views of the Champs-Élysées. Worth every euro for a special occasion."},
    {"city": "Paris", "type": "hotel", "name": "Generator Paris Hostel", "budget": "budget",
     "review": "Best hostel in Paris near Canal Saint-Martin. Dorm beds from €25/night. Great social atmosphere, perfect for solo backpackers. Clean facilities, rooftop bar with Paris views."},
    {"city": "Paris", "type": "restaurant", "name": "Le Comptoir du Relais", "budget": "mid",
     "review": "Classic French bistro in Saint-Germain. Excellent steak frites €18, amazing French onion soup €12. Always crowded but worth the wait. Authentic Parisian experience for couples."},
    {"city": "Paris", "type": "restaurant", "name": "Café de Flore", "budget": "mid",
     "review": "Iconic literary café on Boulevard Saint-Germain. Croque monsieur €14, café au lait €6. Historic atmosphere, famous for Sartre and de Beauvoir. Great for people watching."},
    {"city": "Paris", "type": "restaurant", "name": "L'As du Fallafel", "budget": "budget",
     "review": "Best falafel in Paris in the Jewish Quarter. Only €7 for a giant falafel wrap. Vegetarian friendly, huge portions. Queue moves fast. Budget travelers love this place."},
    {"city": "Paris", "type": "restaurant", "name": "Le Jules Verne", "budget": "luxury",
     "review": "Michelin-starred restaurant inside the Eiffel Tower. Tasting menu from €230. Breathtaking views of Paris. Perfect for romantic dinner proposals. Book months in advance."},
    {"city": "Paris", "type": "restaurant", "name": "Breizh Café", "budget": "mid",
     "review": "Outstanding Breton crêperie in Le Marais. Galettes from €12, sweet crêpes from €8. Organic ingredients, cider from Brittany. Family-friendly atmosphere with kids menu."},
    {"city": "Paris", "type": "attraction", "name": "Eiffel Tower", "budget": "all",
     "review": "Must-see iconic landmark. Summit ticket €29.40, second floor €18.10. Go at sunset for magical views. Book online in advance to skip queues. Romantic for couples."},
    {"city": "Paris", "type": "attraction", "name": "Louvre Museum", "budget": "all",
     "review": "World's largest art museum. Ticket €17, free under 18. See Mona Lisa, Venus de Milo, Winged Victory. Arrive early, allow 3-4 hours minimum. Family-friendly audio guides available."},
    {"city": "Paris", "type": "attraction", "name": "Musée d'Orsay", "budget": "all",
     "review": "Impressionist masterpieces in a stunning former railway station. Ticket €16. Monet, Renoir, Van Gogh collections are world-class. Less crowded than Louvre. Perfect for art lovers."},
    {"city": "Paris", "type": "attraction", "name": "Montmartre & Sacré-Cœur", "budget": "budget",
     "review": "Free to visit the basilica, beautiful hilltop neighborhood. Street artists in Place du Tertre. Amazing panoramic views of Paris. Best explored on foot, romantic at sunset."},
    {"city": "Paris", "type": "attraction", "name": "Palace of Versailles", "budget": "mid",
     "review": "Stunning royal palace 30 min from Paris. Passport €21.50. Gardens free on weekdays. Allow full day. Hall of Mirrors is breathtaking. Book early morning to avoid crowds."},
    {"city": "Paris", "type": "tip", "name": "Paris Budget Tips", "budget": "budget",
     "review": "Paris tips: Metro day pass €8.65, Paris Museum Pass €52/2days covers 60+ museums. Picnic at Champ de Mars with baguette, cheese, wine from local shops costs €10 total. Free Seine river banks walking."},

    # ── LONDON ─────────────────────────────────────────────────────────────────
    {"city": "London", "type": "hotel", "name": "The Hoxton Shoreditch", "budget": "mid",
     "review": "Trendy hotel in hipster Shoreditch neighborhood. Rooms from £150/night. Great rooftop bar, excellent breakfast. Easy tube access to everywhere. Loved by young couples and creatives."},
    {"city": "London", "type": "hotel", "name": "YHA London Central", "budget": "budget",
     "review": "Best value hostel near Oxford Street. Dorms from £25/night, private rooms from £70. Clean facilities, great common areas. Perfect location for budget travelers and solo adventurers."},
    {"city": "London", "type": "hotel", "name": "The Savoy", "budget": "luxury",
     "review": "Legendary 5-star hotel on the Strand since 1889. Rooms from £600/night. Afternoon tea in Thames Foyer is magical at £65/person. Impeccable service, unmatched London luxury experience."},
    {"city": "London", "type": "restaurant", "name": "Dishoom King's Cross", "budget": "mid",
     "review": "Brilliant Bombay-style café in London. Black dal is legendary, bacon naan roll for breakfast £9. Expect queues but they're worth it. Great for groups and families, buzzing atmosphere."},
    {"city": "London", "type": "restaurant", "name": "Borough Market", "budget": "budget",
     "review": "London's greatest food market under London Bridge. Street food from £5-12. Neal's Yard cheese, Monmouth Coffee, fresh bread, international street food. Brilliant for solo foodies and couples."},
    {"city": "London", "type": "restaurant", "name": "Gordon Ramsay Restaurant", "budget": "luxury",
     "review": "Three Michelin stars in Chelsea. Tasting menu £175. Impeccable French cuisine, exceptional wine list. Special occasion dining. Reserve well in advance. Dress code smart casual."},
    {"city": "London", "type": "restaurant", "name": "Flat Iron Covent Garden", "budget": "budget",
     "review": "Amazing flat iron steaks for only £12 in Covent Garden. Simple concept done perfectly. Free popcorn while you wait. No reservations so arrive early. Budget beef lovers paradise."},
    {"city": "London", "type": "attraction", "name": "British Museum", "budget": "budget",
     "review": "One of world's greatest museums, completely FREE. Rosetta Stone, Egyptian mummies, Greek Parthenon sculptures. Allow 3-4 hours. Perfect for families and history lovers. Great museum shop."},
    {"city": "London", "type": "attraction", "name": "Tower of London", "budget": "mid",
     "review": "Historic fortress with Crown Jewels. Adult ticket £34.80. Yeoman Warder tours are brilliant. Allow 2-3 hours. Kids love the armour displays. Book online to save and skip queues."},
    {"city": "London", "type": "attraction", "name": "National Gallery", "budget": "budget",
     "review": "FREE world-class art gallery on Trafalgar Square. Van Gogh Sunflowers, Monet, da Vinci, Turner. Special exhibitions extra charge. Stunning building. Perfect rainy day activity for all."},
    {"city": "London", "type": "attraction", "name": "Hyde Park & Kensington Gardens", "budget": "budget",
     "review": "Beautiful free green space in central London. Serpentine Gallery free. Diana Memorial Playground for kids. Rowing boats for hire £15/hour. Perfect for picnics, cycling, running."},
    {"city": "London", "type": "tip", "name": "London Budget Tips", "budget": "budget",
     "review": "London tips: Oyster card for cheaper tube fares. Many world-class museums are FREE (British Museum, V&A, Natural History, National Gallery, Tate Modern). Thames riverside walk free and beautiful."},

    # ── BARCELONA ──────────────────────────────────────────────────────────────
    {"city": "Barcelona", "type": "hotel", "name": "Casa Camper Barcelona", "budget": "mid",
     "review": "Unique design hotel in El Raval. Rooms from €140/night. Free 24-hour snack room, no restaurant surcharge. Eco-friendly ethos. Central location walking distance to La Boqueria and MACBA."},
    {"city": "Barcelona", "type": "hotel", "name": "Generator Barcelona", "budget": "budget",
     "review": "Best hostel near Passeig de Gràcia. Dorms from €20/night. Rooftop pool in summer, amazing views. Social events, bar on site. Top choice for young solo travelers and backpackers."},
    {"city": "Barcelona", "type": "hotel", "name": "Hotel Arts Barcelona", "budget": "luxury",
     "review": "Iconic 5-star tower hotel on the beach. Rooms from €350/night. Two Michelin-starred restaurant, stunning pool. Celebrity favorite. Best views of Mediterranean from upper floors."},
    {"city": "Barcelona", "type": "restaurant", "name": "Bar Cañete", "budget": "mid",
     "review": "Outstanding tapas bar in El Raval. Croquetas €3 each (best in Barcelona), jamón ibérico €18, patatas bravas €8. Standing bar or sit down. Locals and tourists mix perfectly here."},
    {"city": "Barcelona", "type": "restaurant", "name": "La Boqueria Market", "budget": "budget",
     "review": "Famous covered market on Las Ramblas. Fresh fruit juices €3, jamón sandwiches €5, fresh seafood, local cheeses. Arrive early to avoid tourist crowds. Best budget food experience in Barcelona."},
    {"city": "Barcelona", "type": "restaurant", "name": "Tickets", "budget": "luxury",
     "review": "Albert Adrià's avant-garde tapas bar near Paral·lel. Creative tasting menu €100+. Molecular gastronomy and playful Catalan cuisine. Book months ahead online. Unforgettable foodie experience."},
    {"city": "Barcelona", "type": "restaurant", "name": "Cervecería Catalana", "budget": "mid",
     "review": "Perfect tapas on Carrer Mallorca in Eixample. Patatas bravas €7, grilled vegetables €9, seafood tapas from €8. Always busy, no reservations. Arrive at 1pm or 8pm to get a table."},
    {"city": "Barcelona", "type": "attraction", "name": "Sagrada Família", "budget": "mid",
     "review": "Gaudí's masterpiece cathedral still under construction. Tickets €26-40 with tower access. Absolutely stunning interior with colored light. Book online weeks ahead. Allow 2 hours minimum."},
    {"city": "Barcelona", "type": "attraction", "name": "Park Güell", "budget": "mid",
     "review": "Gaudí's colorful park with famous mosaic terrace. Monumental Zone ticket €10, book online. Go early morning for fewer crowds. Panoramic city views. Free areas surrounding ticketed zone."},
    {"city": "Barcelona", "type": "attraction", "name": "Camp Nou", "budget": "mid",
     "review": "FC Barcelona's legendary stadium. Museum ticket €28 with stadium tour. Must for football fans. Interactive museum tells Club history. Gift shop has everything. Book tour online."},
    {"city": "Barcelona", "type": "attraction", "name": "Barceloneta Beach", "budget": "budget",
     "review": "Free urban beach in Barcelona. Volleyball courts, beach bars (chiringuitos), clear Mediterranean water. Very busy in summer. Watch sunset with cava from a local shop. Perfect free activity."},
    {"city": "Barcelona", "type": "tip", "name": "Barcelona Budget Tips", "budget": "budget",
     "review": "Barcelona tips: T-Casual metro card (10 trips €12.15) saves money. Menu del día lunch special €10-15 at local restaurants is excellent value. Many museums free on Sunday afternoons. Picnic at Park Güell."},

    # ── MADRID ─────────────────────────────────────────────────────────────────
    {"city": "Madrid", "type": "hotel", "name": "Only YOU Hotel Atocha", "budget": "mid",
     "review": "Stylish boutique hotel near Atocha station. Rooms from €100/night. Rooftop pool in summer, fantastic breakfast. Walking distance to Reina Sofía and Prado museums. Perfect base for culture lovers."},
    {"city": "Madrid", "type": "hotel", "name": "Hostel One Madrid", "budget": "budget",
     "review": "Award-winning social hostel near Gran Vía. Dorms from €18/night. Free breakfast, free dinners with hostel drink. Regular pub crawls and social events. Solo travelers rave about community feel."},
    {"city": "Madrid", "type": "restaurant", "name": "Sobrino de Botín", "budget": "mid",
     "review": "World's oldest restaurant (1725) in La Latina. Cochinillo (suckling pig) €28, cordero asado €24. In Guinness World Records. Classic Castilian cuisine. Book in advance, especially on weekends."},
    {"city": "Madrid", "type": "restaurant", "name": "Mercado de San Miguel", "budget": "mid",
     "review": "Beautiful iron and glass market near Plaza Mayor. Tapas from €2-8 each pintxo. Vermouth, fresh seafood, Spanish cheeses. Can get pricey but atmosphere is wonderful. Tourist-friendly but authentic."},
    {"city": "Madrid", "type": "restaurant", "name": "Casa Revuelta", "budget": "budget",
     "review": "Legendary hole-in-the-wall bar in La Latina. Best bacalao (cod) fritters in Madrid at €1.80 each. Standing room only, cash only. Order a caña beer with your tapas. Local institution since 1966."},
    {"city": "Madrid", "type": "attraction", "name": "Prado Museum", "budget": "mid",
     "review": "Spain's greatest art museum. Tickets €15, free evenings 6-8pm. Velázquez Las Meninas, Goya Black Paintings, El Greco. Allow 3-4 hours. Book free evening slot online in advance."},
    {"city": "Madrid", "type": "attraction", "name": "Reina Sofía Museum", "budget": "mid",
     "review": "Modern art museum with Picasso's Guernica. Ticket €12, free Mon/Wed-Sat evenings and Sundays until 2:30pm. Dalí, Miró collections also excellent. Free entry times get crowded."},
    {"city": "Madrid", "type": "attraction", "name": "Retiro Park", "budget": "budget",
     "review": "Madrid's magnificent free park. Row boats on lake €6/hour. Crystal Palace free gallery inside. Sunday outdoor performances often free. Families, runners, couples all love this green oasis."},
    {"city": "Madrid", "type": "tip", "name": "Madrid Budget Tips", "budget": "budget",
     "review": "Madrid tips: Tapas culture means free food with drinks at many bars in La Latina. Lunch Menu del Día €10-13 is best value. Metro 10-trip card €12.20. Many major museums have free entry times."},

    # ── NEW YORK ────────────────────────────────────────────────────────────────
    {"city": "New York", "type": "hotel", "name": "Pod 51 Hotel", "budget": "budget",
     "review": "Affordable micro-room hotel in Midtown. Rooms from $100/night. Small but smart design, great rooftop views. Perfect location walking to Grand Central and Times Square. No frills but clean and modern."},
    {"city": "New York", "type": "hotel", "name": "The High Line Hotel", "budget": "mid",
     "review": "Charming boutique hotel in Chelsea. Rooms from $250/night. Beautiful courtyard, close to High Line park. Stylish design, excellent service. Popular with design-conscious couples and foodies."},
    {"city": "New York", "type": "hotel", "name": "The Plaza Hotel", "budget": "luxury",
     "review": "Legendary 5-star hotel facing Central Park since 1907. Rooms from $1000/night. Afternoon tea in Palm Court, iconic Oak Room bar. Pure New York luxury and history. Special occasion splurge."},
    {"city": "New York", "type": "restaurant", "name": "Katz's Delicatessen", "budget": "budget",
     "review": "NYC institution since 1888 on the Lower East Side. Pastrami sandwich $28 (huge, shareable). Made famous by When Harry Met Sally. Cash preferred. Long lines move fast. Essential New York food experience."},
    {"city": "New York", "type": "restaurant", "name": "Joe's Pizza", "budget": "budget",
     "review": "Best classic New York slice in Greenwich Village. Single slice $3-4, large pie $16. Thin crust, perfect cheese pull, crispy bottom. Cash only. The benchmark for NYC pizza. Quick and cheap."},
    {"city": "New York", "type": "restaurant", "name": "Le Bernardin", "budget": "luxury",
     "review": "Eric Ripert's three Michelin-star seafood temple in Midtown. Tasting menu $195. Best seafood dining in NYC, possibly in USA. Business formal dress code. Exceptional wine pairings available."},
    {"city": "New York", "type": "restaurant", "name": "Smorgasburg Williamsburg", "budget": "budget",
     "review": "Massive outdoor food market in Williamsburg on weekends. 100+ local vendors, dishes $5-15. Ramen burgers, lobster rolls, international street food. Free admission. Brooklyn food culture at its best."},
    {"city": "New York", "type": "attraction", "name": "Central Park", "budget": "budget",
     "review": "843 acres of free urban paradise. Rowboat rentals $15-20/hour, bike rentals nearby. Strawberry Fields, Bethesda Fountain, Sheep Meadow. Free concerts in summer. Perfect for families and all travelers."},
    {"city": "New York", "type": "attraction", "name": "Metropolitan Museum of Art", "budget": "mid",
     "review": "World-class art museum on Fifth Avenue. Suggested admission $30 (technically pay what you wish for NY residents). Egyptian Temple of Dendur, Impressionist galleries, American Wing. Allow full day."},
    {"city": "New York", "type": "attraction", "name": "High Line Park", "budget": "budget",
     "review": "Free elevated park built on old railway line in Chelsea. Great views of Hudson River and city skyline. Art installations, seasonal planting. Perfect for couples and solo walkers. Open dawn to dusk."},
    {"city": "New York", "type": "attraction", "name": "Brooklyn Bridge", "budget": "budget",
     "review": "Walk across the iconic bridge for free - takes about 30 minutes one way. Beautiful views of Manhattan skyline and East River. Visit from DUMBO side for best bridge photos. Go at sunrise or sunset."},
    {"city": "New York", "type": "tip", "name": "New York Budget Tips", "budget": "budget",
     "review": "NYC tips: 7-day MetroCard $34 unlimited subway. Free Staten Island Ferry has best Statue of Liberty views. NYC CityPASS $142 covers top 5 attractions. Happy hour drinks from 4-7pm at many bars. Free outdoor concerts in summer."},

    # ── NEW DELHI ───────────────────────────────────────────────────────────────
    {"city": "New Delhi", "type": "hotel", "name": "The Imperial New Delhi", "budget": "luxury",
     "review": "Historic 5-star colonial hotel in Connaught Place. Rooms from $300/night. Stunning 1930s heritage architecture, outstanding service, excellent restaurants. Delhi's most iconic luxury address for decades."},
    {"city": "New Delhi", "type": "hotel", "name": "Zostel Delhi", "budget": "budget",
     "review": "Best budget hostel in Paharganj near New Delhi station. Dorms from ₹500/night (€6). Social atmosphere, helpful staff for travel advice. Perfect base for exploring Old Delhi and onward travel."},
    {"city": "New Delhi", "type": "hotel", "name": "Lutyens Bungalow", "budget": "mid",
     "review": "Charming colonial bungalow hotel in Lutyens Delhi. Rooms from $80/night. Beautiful garden, home-cooked Indian meals. Quiet and authentic, feels like staying in a private home. Excellent hospitality."},
    {"city": "New Delhi", "type": "restaurant", "name": "Karim's Jama Masjid", "budget": "budget",
     "review": "Legendary Mughlai restaurant near Jama Masjid in Old Delhi since 1913. Mutton korma ₹250, seekh kebabs ₹200, chicken jahangiri ₹300. Authentic flavors unchanged for over 100 years. Cash only."},
    {"city": "New Delhi", "type": "restaurant", "name": "Indian Accent", "budget": "luxury",
     "review": "Best fine dining in Delhi, consistently on Asia's 50 Best list. Modern Indian tasting menu ₹3500-5000. Chef Manish Mehrotra's creative cuisine is extraordinary. Book weeks ahead. Phenomenal wine list."},
    {"city": "New Delhi", "type": "restaurant", "name": "Paranthe Wali Gali Chandni Chowk", "budget": "budget",
     "review": "Famous alley in Chandni Chowk with only paratha restaurants since 1875. Stuffed parathas with various fillings ₹80-150. Mango, banana, khoya, and many vegetable fillings. Unique Old Delhi street food experience."},
    {"city": "New Delhi", "type": "restaurant", "name": "Bukhara ITC Maurya", "budget": "luxury",
     "review": "World-famous North Indian restaurant at ITC Maurya, rated among world's best. Dal Bukhara ₹1200, tandoori meats exceptional. Clinton, Obama, Dalai Lama have dined here. Rustic atmosphere, outstanding food."},
    {"city": "New Delhi", "type": "attraction", "name": "Taj Mahal Agra", "budget": "mid",
     "review": "Wonder of the World, 2.5 hours from Delhi by train or car. Foreign tourist ticket ₹1300. Go at sunrise for magical views. Book Gatimaan Express train from New Delhi station. Allow half day minimum."},
    {"city": "New Delhi", "type": "attraction", "name": "Humayun's Tomb", "budget": "budget",
     "review": "Magnificent Mughal mausoleum that inspired the Taj Mahal. Ticket ₹600 for foreigners. Beautiful Persian gardens, stunning architecture. Less crowded than Taj but equally impressive. UNESCO World Heritage Site."},
    {"city": "New Delhi", "type": "attraction", "name": "Qutub Minar", "budget": "budget",
     "review": "World's tallest brick minaret at 73m. Ticket ₹600. UNESCO World Heritage Site in South Delhi. Beautiful Mughal garden complex, Iron Pillar mystery. Less touristic than other sites. Worth half day."},
    {"city": "New Delhi", "type": "attraction", "name": "Chandni Chowk Old Delhi", "budget": "budget",
     "review": "Chaotic, colorful 17th-century market bazaar. Free to explore. Spice market, silver market, electronics, textiles, street food. Cycle rickshaw tour ₹300 is best way to explore. Visit Jama Masjid nearby."},
    {"city": "New Delhi", "type": "tip", "name": "New Delhi Budget Tips", "budget": "budget",
     "review": "Delhi tips: Delhi Metro Day card ₹200 covers most sights. Ola/Uber rickshaw cheaper than taxis. Eat thali meals at local dhabas for ₹80-150. Carry cash as many local places don't accept cards. Haggle at Dilli Haat market."},
]


def build_dataframe():
    rows = []
    for item in SAMPLE_REVIEWS:
        text = (
            f"City: {item['city']} | Type: {item['type']} | "
            f"Name: {item['name']} | Budget level: {item['budget']} | "
            f"Review: {item['review']}"
        )
        rows.append({
            "city": item["city"],
            "type": item["type"],
            "name": item["name"],
            "budget": item["budget"],
            "text": text,
        })
    return pd.DataFrame(rows)


def ingest():
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Building travel review dataset...")
    df = build_dataframe()
    df.to_csv("data/travel_reviews.csv", index=False)
    print(f"  Saved {len(df)} reviews to data/travel_reviews.csv")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(
        texts=df["text"].tolist(),
        metadatas=df[["city", "type", "name", "budget"]].to_dict("records"),
    )
    print(f"  Split into {len(docs)} chunks")

    print("Loading embedding model (downloads ~80MB on first run)...")
    embeddings = ChromaDefaultEmbeddings()

    print("Writing to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    count = vectorstore._collection.count()
    print(f"Done! ChromaDB now has {count} vectors.")
    return vectorstore


if __name__ == "__main__":
    ingest()