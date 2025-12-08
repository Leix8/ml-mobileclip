prompt_components = {
    "templates": [
        "{s} {a} {sc} {o}",
        "{a} {sc} {o}",
    ],
    "test_scene": {
        "subjects": ["a person", "a cat"],
        "actions": ["is running", "is jumping"],
        "scenarios": ["in the park", "on the beach"],
        "objects": ["with a ball", "with a frisbee"]
    },
    "pet_scene": {
        "subjects": [
            "a dog", "a cat", "a puppy", "a kitten", "two dogs", "two cats",
            "a corgi", "a husky", "a golden retriever", "a small dog", 
            "a large dog", "a small cat", "a pet", "a domestic animal"
        ],
        "actions": [
            "is running", "is jumping", "is walking", "is chasing", "is rolling",
            "is playing", "is fetching", "is tugging a toy", "is biting", 
            "is scratching", "is licking its paw", "is sniffing something", 
            "is shaking its body", "is barking", "is meowing", "is wagging its tail", 
            "is stretching", "is yawning", "is curious", "is watching something", 
            "is sleeping", "is lying down", "is sitting", "is resting", 
            "is staying still"
        ],
        "scenarios": [
            "on the grass", "in the park", "on the beach", "in the living room", 
            "in the bedroom", "in the kitchen", "in the backyard", "in the garden", 
            "on the sofa", "on the bed", "by the window", "on the floor", 
            "in the yard", "at home", "outdoors", "under the table"
        ],
        "objects": [
            "with a ball", "with a frisbee", "with a toy", "with a stick", 
            "with a bone", "with a rope", "with a plush toy", "with a pillow", 
            "with food", "with a bowl", "chasing another pet", 
            "playing with a person", "looking at the camera", "being brushed", 
            "taking a bath", "wearing a collar", "next to its owner"
        ]
    },
    "family_scene": {
        "subjects": [
            "a man", "a woman", "a child", "two children", "a baby", 
            "a teenager", "an elderly man", "an elderly woman", 
            "a father and child", "a mother and child", "two siblings", 
            "grandparents with grandchildren", "a family with a pet", 
            "the whole family"
        ],
        "actions": [
            "is laughing", "is smiling at the camera", "is hugging", 
            "is playing together", "is running", "is jumping", 
            "is holding hands", "is cooking", "is eating together", 
            "is celebrating a birthday", "is opening gifts", 
            "is reading together", "is watching TV", "is taking a selfie", 
            "is posing for a photo", "is walking together", 
            "is playing with the pet", "is decorating for a holiday"
        ],
        "scenarios": [
            "in the living room", "on the couch", "at the dining table", 
            "in the kitchen", "in the backyard", "in the park", 
            "at a playground", "at the beach", "on a city street", 
            "in a bedroom", "outside the house", "at a birthday party", 
            "at a family gathering", "on vacation outdoors", 
            "at a school event"
        ],
        "objects": [
            "with a ball", "with a toy", "with a book", 
            "with a tablet or phone", "with a TV in the background", 
            "with food on the table", "with a birthday cake", 
            "with balloons", "with wrapped gifts", "with a stroller", 
            "with a bicycle", "with a picnic blanket", 
            "with the family pet", "with party decorations", 
            "with holiday decorations"
        ]
    },
    "daily_life_scene": {
        "subjects": [
            "a man", "a woman", "a young adult", "a teenager", 
            "a child", "a baby", "an elderly man", "an elderly woman", 
            "a couple", "a group of friends", "a coworker", 
            "a family", "a student", "a person walking a dog", 
            "a person riding a bicycle"
        ],
        "actions": [
            "is talking on the phone", "is drinking coffee", 
            "is typing on a laptop", "is cooking", "is reading a book", 
            "is writing or drawing", "is waiting at a bus stop", 
            "is walking across the street", "is jogging or exercising", 
            "is taking a photo", "is shopping for groceries", 
            "is doing laundry", "is cleaning the room", 
            "is feeding a pet", "is relaxing on the couch"
        ],
        "scenarios": [
            "in the kitchen", "in the living room", "in a bedroom", 
            "at a coffee shop", "at the office", "in a park", 
            "on a city street", "at the bus stop", "in a classroom", 
            "in a supermarket", "at the gym", "on public transportation", 
            "on the balcony", "in the backyard", "at a restaurant table"
        ],
        "objects": [
            "with a phone", "with a laptop", "with a tablet", 
            "with a cup of coffee", "with a newspaper or magazine", 
            "with a backpack", "with shopping bags", "with earphones", 
            "with gym equipment", "with cleaning tools", 
            "with books and notebooks", "with a camera", 
            "with food or drinks", "with a pet", 
            "with groceries or fruits"
        ]
    },
    "sport_and_workout_scene": {
        "subjects": [
            "a male athlete", "a female athlete", "a runner", "a sprinter", 
            "a cyclist", "a swimmer", "a soccer player", "a basketball player", 
            "a tennis player", "a boxer", "a weightlifter", "a yoga instructor", 
            "a coach celebrating", "a cheering teammate", "a team celebrating a victory"
        ],
        "actions": [
            "is sprinting at full speed", "is jumping high in the air", 
            "is diving into the pool", "is scoring a goal", 
            "is shooting a basketball", "is swinging a racket mid-serve", 
            "is punching hard during training", "is lifting heavy weights overhead", 
            "is crossing the finish line", "is celebrating a win", 
            "is shouting in excitement", "is clapping or cheering", 
            "is stretching before exercise", "is exhausted after a match", 
            "is posing proudly with a medal"
        ],
        "scenarios": [
            "on a running track during a race", "on a soccer field during a match", 
            "on a basketball court in mid-game", "on a tennis court during a rally", 
            "at a swimming pool during competition", "inside a gym during workout", 
            "on a yoga mat in natural light", "on a mountain trail during a run", 
            "on a beach during morning workout", "in a boxing ring under bright lights", 
            "at a stadium filled with crowd", "on a podium receiving a medal", 
            "in an outdoor sports event at sunset", "in a training field under rain", 
            "in a locker room post-game moment"
        ],
        "objects": [
            "with a ball mid-air", "with a tennis racket in motion", 
            "with a basketball in hand", "with a dumbbell or barbell", 
            "with a skipping rope mid-jump", "with boxing gloves", 
            "with a bicycle helmet on", "with a yoga mat", 
            "with a stopwatch or timer", "with sweat and towel", 
            "with teammates hugging", "with a trophy or medal", 
            "with sports shoes and uniform", "with gym equipment in the background", 
            "with water splashing around"
        ]
    }, 
    "fishing_scene": {
        "subjects": [
            "an angler", "a fisherman", "a fisherwoman", "a person fishing",
            "two people fishing", "a fishing partner", "a guide and a client",
            "a person holding a fishing rod", "a person wearing waders",
            "a person on a boat", "a person by the shoreline", 
            "a person standing in shallow water",
            "a first-person perspective of the angler", 
            "a GoPro first-person view", "a drone view of an angler"
        ],
        "actions": [
            "is casting a line", "is reeling in quickly", "is reeling smoothly",
            "is setting the hook", "is fighting a fish", 
            "is struggling with a strong pull",
            "is landing a fish", "is lifting the fish out of the water",
            "is releasing a fish", "is holding a fish up for the camera",
            "is reacting excitedly", "is shouting in excitement",
            "is carefully unhooking a fish",
            "is watching the bobber dip", "is adjusting the reel drag",
            "is waiting patiently", "is checking the bait",
            "is pulling the rod upward", "is tightening the fishing line",
            
            "is experiencing a sudden rod bend", 
            "is hit by a sudden bite",
            "is reacting to a splash",
            
            "is moving the boat", "is paddling", "is positioning the boat"
        ],
        "scenarios": [
            "on a lake", "on a river", "on a calm pond", "on the ocean",
            "on a fishing boat", "on a kayak", "on a small skiff",
            "from a pier", "from a dock", "from the beach",
            "in shallow water", "in waders", "standing in the river current",
            "on a rocky shoreline", "near tall grass", "near reeds",
            "at sunrise", "at sunset", "on a cloudy day",
            "in windy conditions", "in bright sunlight",
            
            "from a first-person camera", "from a chest-mounted GoPro",
            "from a head-mounted camera", "from a drone overhead",
            
            "during a strong splash", "during a fast reel-in",
            "during the hook-set moment", "during a rod-bending fight"
        ],
        "objects": [
            "with a fishing rod", "with a spinning reel", "with a baitcaster",
            "with a fly rod", "with a lure", "with a baited hook",
            "with a bobber", "with a net", "with a tackle box",
            "holding a fish", "with a fish jumping", "with a fish splashing water",
            "with a large fish", "with a small fish", "with a fish near the surface",
            "with the line tight", "with the rod bending heavily",
            "with water splashing upward", "with the hook visible",
            "with a hook-set moment", "with the lure visible in water",
            
            "with a boat motor", "with paddles", "with a cooler",
            "with fishing gear scattered nearby"
        ]
    }, 
    "basketball_scene": {
        "subjects": [
            "a basketball player", "a person playing basketball", 
            "a person dribbling", "a person shooting a basketball",
            "a defender guarding closely", "an offensive player driving forward",
            "two people playing one-on-one", "a group playing pick-up basketball",
            "a coach and a player", "a small group doing drills",
            "a family playing basketball", "kids playing basketball",
            "a high school basketball team", "a college basketball player",
            "a referee on the court", "a bench of teammates cheering",
            
            "a person wearing a jersey", "a person wearing sportswear",
            "a person jumping for a shot", "a person dunking",
            
            "a first-person perspective of dribbling", 
            "a GoPro first-person view of shooting",
            "a drone view of a full court"
        ],
        
        "actions": [
            "is dribbling quickly", "is dribbling between the legs",
            "is dribbling behind the back", "is crossing over an opponent",
            "is driving toward the basket", "is making a fast break",
            "is passing the ball", "is performing a no-look pass",
            "is shooting a three-pointer", "is shooting a jump shot",
            "is attempting a layup", "is performing a reverse layup",
            "is dunking the ball", "is attempting an alley-oop",
            "is blocking a shot", "is contesting a shot",
            "is stealing the ball", "is diving for a loose ball",
            "is grabbing a rebound", "is boxing out an opponent",
            
            "is celebrating after scoring", "is shouting in excitement",
            "is reacting to a missed shot", "is calling for the ball",
            "is signaling a play", "is defending tightly",
            
            "is practicing dribbling drills", "is practicing shooting drills",
            "is practicing free throws", "is running conditioning drills",
            
            "is waiting on the free-throw line", 
            "is watching a teammate shoot",
            "is checking the shot clock",
            
            "is running back on defense", "is hustling to the ball"
        ],
        
        "scenarios": [
            "on an indoor basketball court", "on an outdoor court",
            "in a school gym", "on a neighborhood court",
            "in a driveway at home", "in a public park",
            "in a college arena", "in a large stadium",
            "on a small half-court", "on a full-sized court",
            
            "during a pick-up game", "during a practice session",
            "during a team scrimmage", "during a school match",
            "during a competitive tournament", "during a championship game",
            
            "at sunset", "at night under court lights", 
            "in bright daylight", "in a noisy packed gym",
            
            "from a first-person camera", "from a chest GoPro",
            "from a head-mounted camera", "from a drone view",
            
            "during a fast break", "during a blocked shot",
            "during a game-winning shot attempt",
            "during a steal and transition play",
            "during a rebound battle", "during an intense defensive play"
        ],
        
        "objects": [
            "with a basketball", "with a hoop", "with a backboard",
            "with a net swishing", "with a loose ball on the floor",
            "with the ball mid-air", "with a scoreboard visible",
            "with a shot clock counting down", "with cones for drills",
            
            "with a player wearing a jersey", "with matching uniforms",
            "with a defender's hand extended", "with both feet off the ground",
            "with the ball about to hit the rim", "with the ball bouncing",
            "with sneakers squeaking on the floor", "with chalk dust or sweat flying",
            
            "with a crowd cheering", "with a bench celebrating",
            "with a coach giving instructions", "with a referee signaling",
            
            "with a clean backboard", "with a glass backboard reflection",
            "with a chain net", "with an outdoor metal hoop",
            
            "with the ball spinning", "with a swish moment",
            "with the net moving", "with the rim shaking after a dunk"
        ]
    },
    "soccer_scene": {

        "subjects": [
            "a soccer player", "a person playing soccer",
            "a striker running forward", "a midfielder controlling the ball",
            "a defender blocking a shot", "a goalkeeper diving for a save",
            "two people playing soccer", "kids playing soccer",
            "a group practicing passing drills",
            "a youth soccer team", "a high school soccer team",
            "a family playing soccer in the yard",
            "a coach giving instructions", "a referee on the field",
            "a crowd cheering from the sideline",
            
            "a person wearing cleats", "a player in a jersey",
            "a person sprinting", "a person juggling the ball",
            
            "a first-person dribbling view", 
            "a GoPro chest-mounted view",
            "a drone view over a soccer field"
        ],

        "actions": [
            "is dribbling quickly", "is dribbling past a defender",
            "is maneuvering the ball with close control",
            "is passing the ball", "is making a long pass",
            "is crossing the ball into the box", 
            "is taking a powerful shot", "is shooting a volley",
            "is attempting a bicycle kick", "is taking a penalty kick",
            "is taking a free kick", "is curling the ball toward the goal",
            "is scoring a goal", "is missing a shot",
            
            "is tackling an opponent", "is sliding for a tackle",
            "is blocking a shot", "is intercepting a pass",
            "is shielding the ball", "is jostling for position",
            
            "is heading the ball", "is chest-trapping the ball",
            "is juggling the ball", "is controlling a high pass",
            
            "is celebrating a goal", "is shouting in excitement",
            "is disappointed after a miss", "is raising hands to teammates",
            
            "is diving for a save", "is grabbing the ball mid-air",
            "is punching the ball away", "is catching a fast shot",
            
            "is practicing dribbling cones", "is practicing shooting drills",
            "is performing warm-up exercises", "is stretching on the sideline",

            "is sprinting back to defend", "is running into open space"
        ],

        "scenarios": [
            "on a grass field", "on a turf field", "on a dusty outdoor field",
            "in a school field", "in a stadium", "on a community park field",
            "in a backyard", "on a small training pitch",
            "on a rainy field", "on a muddy field",
            "on a sunny day", "at sunset", "at night under stadium lights",

            "during a casual game", "during practice drills",
            "during a youth league match", "during a school match",
            "during a competitive tournament", "during a championship final",
            
            "from a drone overhead", "from a sideline camera",
            "from a first-person head cam", "from a goalkeeper POV",

            "during a fast counterattack", "during a corner kick",
            "during a free kick opportunity", "during a one-on-one with the goalkeeper",
            "during a slide tackle", "during a goal celebration",
            "during a penalty shootout", "during a defensive wall setup"
        ],

        "objects": [
            "with a soccer ball", "with a rolling ball",
            "with the ball mid-air", "with a spinning ball",
            "with a goalpost", "with a net shaking from a goal",
            "with corner flags", "with cones for drills",
            "with goalkeeper gloves", "with shin guards",
            
            "with a scoreboard visible", "with cheering fans",
            "with team benches on the side", "with a coach holding a clipboard",
            
            "with grass flying after a kick", "with dirt splashing",
            "with water spray from wet grass",
            
            "with an offside flag", "with a referee whistle",
            
            "with the ball bouncing off the post", 
            "with the ball crossing the goal line",
            "with the net moving after a shot",
            
            "with colorful jerseys", "with cleats digging into the turf"
        ]
    },
    "hiking_scene": {
        "subjects": [
            "a hiker", "a backpacker", "a person hiking alone",
            "two friends hiking", "a group of hikers", 
            "a family hiking together", "a couple on a hike",
            "a person walking with trekking poles", 
            "a person climbing uphill", "a person descending a steep trail",
            "a trail runner", "a nature photographer hiking",
            "a first-person hiking view", "a GoPro chest-mounted hiker",
            "a drone view of hikers on a trail"
        ],

        "actions": [
            "is climbing uphill", "is descending a steep slope",
            "is walking along the trail", "is stepping over rocks",
            "is crossing a stream", "is jumping over a puddle",
            "is navigating switchbacks", "is using trekking poles",
            "is taking in the view", "is pointing at scenery",
            "is drinking water", "is taking a break on a rock",
            "is adjusting their backpack straps", "is checking a map",
            "is tying shoelaces", "is photographing the landscape",
            "is watching wildlife", "is brushing past branches",
            
            "is reacting to a beautiful view", "is celebrating at the summit",
            
            "is hiking at sunrise", "is hiking at sunset",
            "is hiking in fog", "is hiking in light rain"
        ],

        "scenarios": [
            "on a mountain trail", "on a forest trail", "on a rocky path",
            "on a desert trail", "in a national park", "on a riverside trail",
            "through a dense forest", "on an open ridge", 
            "on a snowy path", "on a muddy trail",
            "at a scenic viewpoint", "at the summit", 
            "in a canyon", "near a waterfall",
            
            "during golden hour", "in bright daylight",
            "under cloudy skies", "during misty conditions",
            
            "from a first-person POV", "from a drone overhead"
        ],

        "objects": [
            "with a backpack", "with trekking poles", "with hiking boots",
            "with a hydration pack", "with a water bottle",
            "with a trail map", "with a GPS device",
            "with a rocky trail", "with tree roots on the ground",
            "with dust kicking up", "with leaves covering the trail",
            "with a waterfall nearby", "with mountains in the background",
            
            "with sunlight filtering through trees", 
            "with fog rolling in",
            
            "with a summit sign", "with a trail marker",
            "with wildlife in the distance"
        ]
    },
    "cycling_scene": {
        "subjects": [
            "a cyclist", "a road cyclist", "a mountain biker",
            "a person riding a bike", "a kid riding a bicycle",
            "two cyclists riding together", "a cycling group or peloton",
            "a family cycling", "a bike commuter",
            "a person on an e-bike", "a downhill mountain biker",
            
            "a first-person handlebar perspective", 
            "a GoPro chest-mounted cyclist", 
            "a drone view of cyclists"
        ],

        "actions": [
            "is pedaling fast", "is cruising downhill",
            "is climbing a steep hill", "is standing up on the pedals",
            "is making a sharp turn", "is drifting on gravel",
            "is hitting a jump", "is landing after a jump",
            "is braking hard", "is balancing at low speed",
            "is overtaking another cyclist", "is drafting behind another rider",
            
            "is reacting with excitement", "is celebrating crossing a finish line",
            
            "is riding hands-free", "is signaling a turn",
            "is checking over their shoulder",
            
            "is wiping sweat", "is adjusting their helmet"
        ],

        "scenarios": [
            "on a mountain trail", "on a forest singletrack",
            "on a paved road", "on a bike lane",
            "on a gravel road", "on a dirt trail",
            "along a riverside path", "along a coastal road",
            "in a city street", "in a suburban neighborhood",
            "in a bike park", "in a race event",
            
            "at sunrise", "at sunset", "in bright sunlight",
            "in fog", "under light rain",
            
            "during a fast descent", "during a climb",
            "during a sprint finish", "during a muddy trail ride"
        ],

        "objects": [
            "with a road bike", "with a mountain bike", "with a gravel bike",
            "with an e-bike", "with clipless pedals",
            "with a winding trail", "with a steep climb",
            "with dust or dirt kicking up", "with mud splashing",
            
            "with a helmet", "with cycling gloves", "with sunglasses",
            "with a bike computer", "with a water bottle",
            
            "with a chain spinning", "with wheels in motion",
            "with spokes blurring", "with tires skidding",
            
            "with a scenic background", "with buildings passing by",
            "with a race banner or finish line"
        ]
    },
    "wedding_scene": {
        "subjects": [
            "a bride", "a groom",
            "a couple getting married", "newlyweds celebrating",
            "a flower girl", "a ring bearer",
            "a family posing together", "guests celebrating",
            "a photographer capturing moments",
            
            "a close-up of hands", "a first-person perspective walking down the aisle",
            "a drone view of the wedding venue"
        ],

        "actions": [
            "is walking down the aisle", "is exchanging vows",
            "is exchanging rings", "is holding hands",
            "is kissing at the altar", "is hugging joyfully",
            "is laughing together", "is wiping tears",
            "is dancing together", "is having their first dance",
            "is cutting the wedding cake", "is feeding cake to each other",
            "is throwing the bouquet", "is catching the bouquet",
            
            "is posing for photos", "is signing the marriage certificate",
            
            "is celebrating with friends", "is cheering with guests",
            
            "is adjusting the dress", "is fixing the boutonniere"
        ],

        "scenarios": [
            "at an outdoor wedding", "at an indoor wedding",
            "in a church", "in a garden venue",
            "on a beach", "on a mountain overlook",
            "in a hotel ballroom", "in a rustic barn",
            
            "during the ceremony", "during the reception",
            "during golden hour", "during a night celebration",
            
            "under string lights", "under floral decorations",
            "on a dance floor", "at the altar",
            
            "from a drone overhead", "from a first-person POV"
        ],

        "objects": [
            "with a wedding dress", "with a suit or tuxedo",
            "bridesmaids", "groomsmen", "a wedding party",
            "with a bridal bouquet", "with a boutonniere",
            "with wedding rings", "with a veil flowing",
            "with floral decorations", "with candles",
            "with a wedding cake", "with champagne glasses",
            
            "with confetti falling", "with flower petals on the aisle",
            "with guests applauding", "with fairy lights",
            
            "with a decorated arch", "with a wedding sign",
            "with a guest book", "with a seating chart",
            
            "with a sunset backdrop", "with a scenic venue",
            "with a dance floor setup"
        ]
    },
    "tennis_scene": {
        "subjects": [
            "a tennis player", "a person holding a racket",
            "two players rallying", "a doubles team",
            "a coach training a student", "a family playing tennis",
            "a beginner practicing serves", "a competitive match player",
            "a first-person racket view", "a drone view of a tennis court"
        ],
        "actions": [
            "is serving powerfully", "is performing a slice serve",
            "is hitting a forehand", "is hitting a backhand",
            "is returning a serve", "is rallying quickly",
            "is hitting a volley", "is hitting an overhead smash",
            "is running to the net", "is sliding to reach the ball",
            "is stretching for a wide shot", "is celebrating a winner",
            "is reacting to a missed shot", "is adjusting grip"
        ],
        "scenarios": [
            "on an outdoor court", "on an indoor court",
            "on a hard court", "on a clay court", "on a grass court",
            "during a casual match", "during a lesson",
            "during a competitive tournament",
            "at sunset", "under bright sunlight", "under stadium lights"
        ],
        "objects": [
            "with a tennis racket", "with a tennis ball",
            "with a ball mid-air", "with a net in view",
            "with ball marks on court", "with a scoreboard",
            "with a water bottle", "with a tennis bag",
            "with the ball bouncing", "with chalk dust on clay"
        ]
    },
    "badminton_scene": {
        "subjects": [
            "a badminton player", "a person holding a racket",
            "two players rallying", "a doubles team",
            "a coach training a beginner", "kids playing badminton",
            "a competitive club player", "a family in the backyard"
        ],
        "actions": [
            "is serving a shuttlecock", "is performing a smash",
            "is hitting a drop shot", "is hitting a clear",
            "is lunging forward", "is jumping for a smash",
            "is reacting quickly", "is diving for a save",
            "is celebrating a point", "is retrieving a low shot"
        ],
        "scenarios": [
            "in an indoor sports hall", "on an outdoor court",
            "backyard badminton", "a school gym match",
            "a club tournament", "casual warm-up",
            "under bright lights", "in a crowded gym"
        ],
        "objects": [
            "with a shuttlecock", "with a badminton racket",
            "with a net", "with court lines",
            "with a birdie in mid-air", "with a racket bag",
            "with lightweight shoes", "with overhead lighting reflections"
        ]
    },
    "bowling_scene": {
        "subjects": [
            "a bowler", "a person holding a bowling ball",
            "a group of friends bowling", "a family bowling night",
            "kids bowling with bumpers", "a competitive league bowler"
        ],
        "actions": [
            "is swinging the ball", "is releasing the ball smoothly",
            "is attempting a strike", "is aiming for a spare",
            "is celebrating a strike", "is reacting with excitement",
            "is watching the ball roll", "is adjusting stance"
        ],
        "scenarios": [
            "in a bowling alley", "at a league match",
            "in a neon-lit cosmic bowling session", 
            "at a family entertainment center"
        ],
        "objects": [
            "with a bowling ball", "with pins falling",
            "with a bowling lane", "with score screens overhead",
            "with rental shoes", "with lane arrows",
            "with gutter bumpers", "with a ball return system"
        ]
    },
    "golfing_scene": {
        "subjects": [
            "a golfer", "a person holding a golf club",
            "two friends golfing", "a foursome on a course",
            "a golfer with a caddie", "a family practicing at the range",
            "a first-person swing perspective", "a drone view of a fairway"
        ],
        "actions": [
            "is taking a full swing", "is teeing off",
            "is hitting a long drive", "is performing a chip shot",
            "is putting on the green", "is lining up a putt",
            "is watching the ball flight", "is celebrating a good shot",
            "is raking the sand bunker", "is cleaning the club"
        ],
        "scenarios": [
            "on a golf course fairway", "on the putting green",
            "at a golf driving range", "at a practice bunker",
            "during a sunny round", "at sunrise", "in the late afternoon",
            "during a tournament", "at a resort course"
        ],
        "objects": [
            "with a golf club", "with a golf ball",
            "with tees on the ground", "with a golf cart",
            "with a flagstick", "with a sand bunker",
            "with a water hazard", "with greenside grass",
            "with a scorecard", "with a glove"
        ]
    },
    "table_tennis_scene": {
        "subjects": [
            "a table tennis player", "a person holding a paddle",
            "two players rallying", "a doubles match",
            "a coach teaching a beginner", "kids practicing",
            "a competitive league player"
        ],
        "actions": [
            "is serving the ball", "is performing a smash",
            "is hitting a topspin shot", "is blocking at the table",
            "is looping forehand", "is counterattacking",
            "is diving to save the ball", "is celebrating a point"
        ],
        "scenarios": [
            "in a sports hall", "in a recreation center",
            "in a garage setup", "in a school gym",
            "during practice drills", "during a tournament"
        ],
        "objects": [
            "with a ping pong ball", "with a paddle",
            "with a table tennis table", "with a net",
            "with spin on the ball", "with a scoreboard",
            "with barriers around the table", "with sweat towel"
        ]
    },
    "volleyball_scene": {
        "subjects": [
            "a volleyball player", "two players at the net",
            "a full team", "a beach volleyball pair",
            "a setter", "a hitter", "a libero",
            "a group playing recreational volleyball"
        ],
        "actions": [
            "is serving the ball", "is jumping to spike",
            "is blocking at the net", "is diving to dig",
            "is setting the ball", "is receiving a serve",
            "is celebrating a point", "is signaling a play"
        ],
        "scenarios": [
            "on an indoor court", "on a beach court",
            "in a school gym", "in a sand recreational area",
            "during a competitive match", "during practice",
            "under bright gym lights", "at sunset on the beach"
        ],
        "objects": [
            "with a volleyball", "with a net",
            "with sand flying", "with floor dust",
            "with court lines", "with kneepads",
            "with team uniforms", "with cheering crowd"
        ]
    },
    "gym_scene": {
        "subjects": [
            "a gym-goer", "a weightlifter", "a person training",
            "a personal trainer coaching a client", 
            "a group workout class", "a bodybuilder",
            "a cardio exerciser", "a powerlifter",
            "a person stretching", "a first-person workout POV"
        ],
        "actions": [
            "is lifting weights", "is squatting heavy",
            "is deadlifting", "is bench pressing",
            "is curling dumbbells", "is doing pull-ups",
            "is doing push-ups", "is doing planks",
            "is running on a treadmill", "is rowing",
            "is cycling on a stationary bike", "is adjusting machine settings",
            "is sweating intensely", "is celebrating a PR"
        ],
        "scenarios": [
            "in a gym with free weights", "in a cardio area",
            "in a functional training zone", "in a powerlifting area",
            "in a group class room", "in a garage gym",
            "during strength training", "during cardio",
            "during peak hours", "in an empty gym"
        ],
        "objects": [
            "with dumbbells", "with barbells",
            "with weight plates", "with kettlebells",
            "with resistance bands", "with a treadmill",
            "with a squat rack", "with cable machines",
            "with chalk dust", "with mirrors"
        ]
    },
    "swimming_scene": {
        "subjects": [
            "a swimmer", "a person swimming laps",
            "two swimmers racing", "a family in a pool",
            "a coach teaching kids", "a competitive swimmer",
            "a lifeguard observing", "a first-person underwater view"
        ],
        "actions": [
            "is freestyle swimming", "is doing breaststroke",
            "is doing butterfly stroke", "is doing backstroke",
            "is diving into the water", "is turning at the wall",
            "is kicking underwater", "is racing in lanes",
            "is floating", "is splashing water"
        ],
        "scenarios": [
            "in an indoor pool", "in an outdoor pool",
            "in a competition pool", "in a backyard pool",
            "at a beach", "in a lake", "in a waterpark",
            "during a swim meet", "during practice"
        ],
        "objects": [
            "with swimming goggles", "with a swim cap",
            "with lane lines", "with diving blocks",
            "with splashing water", "with bubbles underwater",
            "with float boards", "with kickboards"
        ]
    },
    "yoga_scene": {
        "subjects": [
            "a yoga practitioner", "a person doing yoga",
            "two people practicing", "a group yoga class",
            "a yoga instructor", "a family doing yoga",
            "a person meditating", "a first-person yoga mat view"
        ],
        "actions": [
            "is holding a pose", "is stretching deeply",
            "is doing downward dog", "is doing warrior pose",
            "is balancing on one leg", "is doing a handstand",
            "is breathing calmly", "is meditating",
            "is transitioning between poses"
        ],
        "scenarios": [
            "in a yoga studio", "in a home yoga space",
            "outdoors in a park", "on a beach",
            "during sunrise yoga", "during a hot yoga class",
            "in a quiet, dim room"
        ],
        "objects": [
            "with a yoga mat", "with yoga blocks",
            "with a strap", "with candles",
            "with natural light", "with indoor plants",
            "with meditation cushions"
        ]
    },
    "skiing_scene": {
        "subjects": [
            "a skier", "a person skiing downhill",
            "a ski instructor with a student",
            "a group skiing together", "a family skiing",
            "a first-person ski goggle POV", "a drone view of skiers"
        ],
        "actions": [
            "is carving turns", "is skiing downhill fast",
            "is jumping off a slope", "is landing a jump",
            "is stopping abruptly", "is skating uphill",
            "is adjusting goggles", "is falling into snow",
            "is celebrating at the bottom of the slope"
        ],
        "scenarios": [
            "on a snowy mountain", "on a ski resort trail",
            "in a beginner area", "in a terrain park",
            "during heavy snow", "during a sunny ski day",
            "at sunset", "on a crowded slope"
        ],
        "objects": [
            "with skis", "with ski poles", "with goggles",
            "with snow spraying", "with a ski lift",
            "with a snowy trail", "with ski boots",
            "with a mountain backdrop"
        ]
    },
    "skating_scene": {
        "subjects": [
            "a skater", "a person ice skating",
            "a person roller skating", "two friends skating",
            "a figure skater", "a hockey skater practicing",
            "kids skating", "a couple holding hands while skating"
        ],
        "actions": [
            "is gliding smoothly", "is spinning", "is jumping",
            "is performing footwork", "is braking sharply",
            "is falling lightly", "is helping someone up",
            "is skating backward", "is practicing routines"
        ],
        "scenarios": [
            "on an ice rink", "on a frozen lake",
            "at a roller skating rink", "on a smooth outdoor path",
            "during a figure skating practice", "during a hockey warm-up",
            "under colorful rink lights"
        ],
        "objects": [
            "with ice skates", "with roller skates",
            "with a hockey stick", "with cones for practice",
            "with reflections on ice", "with skate marks",
            "with rink boards", "with protective pads"
        ]
    },
    "boxing_scene": {
        "subjects": [
            "a boxer", "two boxers sparring",
            "a coach holding pads", "a trainee learning punches",
            "a fitness boxer", "a professional fighter",
            "a person skipping rope", "a person shadowboxing"
        ],
        "actions": [
            "is throwing jabs", "is throwing hooks",
            "is throwing uppercuts", "is slipping punches",
            "is blocking and parrying", "is moving around the ring",
            "is hitting focus mitts", "is hitting the heavy bag",
            "is sweating intensely", "is celebrating a clean hit"
        ],
        "scenarios": [
            "in a boxing gym", "in a training ring",
            "in a competition ring", "during sparring practice",
            "during conditioning drills", "during a pro match",
            "under bright fight lights"
        ],
        "objects": [
            "with boxing gloves", "with hand wraps",
            "with a heavy bag", "with a speed bag",
            "with focus mitts", "with a jump rope",
            "with ring ropes", "with a corner stool"
        ]
    }
}
