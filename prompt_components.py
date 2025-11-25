prompt_components = {
    "templates": [
        "{s} {a} {sc} {o}"
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
    }
}
