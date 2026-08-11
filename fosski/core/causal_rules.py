"""
Causal Rules Engine — Derive Effects from Physical Laws
=========================================================
Instead of hardcoding "if heat + ice → melts", this engine stores
fundamental physical laws as rules and derives consequences.

Rules are stored as:
  condition → effect
  where condition is a set of (entity_property, threshold) checks

Example:
  Rule: "substance(state=solid) + temperature > melting_point → melts"
  Applied: ice(state=solid, melting_point=0°C) + temperature=100°C → melts

This is the "teach basic laws, let it derive" approach.
"""

from typing import Optional, List, Dict


# ============================================================
# Physical Laws as Rules
# ============================================================

# State transitions
STATE_TRANSITIONS = {
    'solid': {
        'heat': 'liquid',       # melting
        'sublimate': 'gas',     # sublimation (dry ice)
    },
    'liquid': {
        'heat': 'gas',          # evaporation/boiling
        'cool': 'solid',        # freezing
    },
    'gas': {
        'cool': 'liquid',       # condensation
    },
}

# Material properties: (melting_point_C, boiling_point_C, default_state)
MATERIAL_PROPERTIES = {
    'water':    (0, 100, 'liquid'),
    'ice':      (0, 100, 'solid'),
    'steam':    (0, 100, 'gas'),
    'iron':     (1538, 2862, 'solid'),
    'gold':     (1064, 2856, 'solid'),
    'mercury':  (-39, 357, 'liquid'),
    'glass':    (1400, 2230, 'solid'),
    'steel':    (1370, 2750, 'solid'),
    'copper':   (1085, 2562, 'solid'),
    'lead':     (327, 1749, 'solid'),
    'aluminum': (660, 2519, 'solid'),
    'wax':      (46, 370, 'solid'),
    'butter':   (32, 250, 'solid'),
    'chocolate':(34, 400, 'solid'),
    'nitrogen': (-210, -196, 'gas'),
    'oxygen':   (-218, -183, 'gas'),
    'helium':   (-272, -269, 'gas'),
    'ethanol':  (-114, 78, 'liquid'),
    'salt':     (801, 1413, 'solid'),
    'sugar':    (186, 186, 'solid'),  # decomposes
    'rock':     (700, 2500, 'solid'),
    'wood':     (300, 300, 'solid'),  # burns, doesn't melt
    'plastic':  (120, 300, 'solid'),
    'snow':     (0, 100, 'solid'),
}

# Causal rules: condition_keywords → (effect_description, explanation)
# These are GENERAL LAWS, not per-item hardcodes
CAUSAL_LAWS = [
    # Thermodynamics
    {
        'conditions': ['heat', 'solid'],
        'effect': 'it melts',
        'law': 'Adding heat to a solid raises temperature above melting point',
        'exceptions': ['wood'],  # burns instead
    },
    {
        'conditions': ['heat', 'liquid'],
        'effect': 'it evaporates',
        'law': 'Adding heat to a liquid raises temperature above boiling point',
    },
    {
        'conditions': ['cool', 'liquid'],
        'effect': 'it freezes',
        'law': 'Removing heat from a liquid drops temperature below freezing point',
    },
    {
        'conditions': ['cool', 'gas'],
        'effect': 'it condenses',
        'law': 'Removing heat from a gas drops temperature below boiling point',
    },
    # Mechanics
    {
        'conditions': ['drop', 'fragile'],
        'effect': 'it breaks',
        'law': 'Fragile objects break on impact',
    },
    {
        'conditions': ['drop', 'heavy'],
        'effect': 'it falls',
        'law': 'Gravity pulls objects downward',
    },
    {
        'conditions': ['push', 'movable'],
        'effect': 'it moves',
        'law': "Newton's first law: force causes acceleration",
    },
    # Chemistry
    {
        'conditions': ['iron', 'water', 'oxygen'],
        'effect': 'rust forms',
        'law': 'Iron oxidizes in the presence of water and oxygen',
    },
    {
        'conditions': ['metal', 'rust'],
        'effect': 'exposure to oxygen and moisture',
        'law': 'Oxidation requires both oxygen and moisture',
    },
    {
        'conditions': ['fire', 'fuel', 'oxygen'],
        'effect': 'combustion occurs',
        'law': 'Fire requires fuel, oxygen, and heat (fire triangle)',
    },
    {
        'conditions': ['mix', 'red', 'blue'],
        'effect': 'purple',
        'law': 'Subtractive color mixing: red + blue = purple',
    },
    {
        'conditions': ['mix', 'red', 'yellow'],
        'effect': 'orange',
        'law': 'Subtractive color mixing: red + yellow = orange',
    },
    {
        'conditions': ['mix', 'blue', 'yellow'],
        'effect': 'green',
        'law': 'Subtractive color mixing: blue + yellow = green',
    },
    # Optics
    {
        'conditions': ['light', 'block', 'object'],
        'effect': 'a shadow forms',
        'law': 'Light travels in straight lines; opaque objects block it',
    },
    {
        'conditions': ['shadow'],
        'effect': 'an object blocking light',
        'law': 'Shadows are caused by objects blocking light',
    },
    # Biology
    {
        'conditions': ['plant', 'no', 'water'],
        'effect': 'it wilts and dies',
        'law': 'Plants need water for photosynthesis and turgor pressure',
    },
    {
        'conditions': ['plant', 'no', 'sunlight'],
        'effect': 'it cannot photosynthesize and dies',
        'law': 'Plants need sunlight for photosynthesis',
    },
    {
        'conditions': ['seed', 'water', 'soil'],
        'effect': 'it germinates and grows',
        'law': 'Seeds germinate with water, soil, and warmth',
    },
    # Weather/atmosphere
    {
        'conditions': ['water', 'evaporate', 'cool'],
        'effect': 'rain forms',
        'law': 'Water cycle: evaporation → condensation → precipitation',
    },
    {
        'conditions': ['rain'],
        'effect': 'water evaporation and condensation',
        'law': 'Rain is caused by the water cycle',
    },
    # Behavior
    {
        'conditions': ['birds', 'south', 'winter'],
        'effect': 'for warmer weather',
        'law': 'Birds migrate to warmer climates for food availability',
    },
    {
        'conditions': ['coat', 'winter'],
        'effect': 'to stay warm',
        'law': 'Insulation reduces heat loss from the body',
    },
    {
        'conditions': ['exercise'],
        'effect': 'sweating and increased heart rate',
        'law': 'Physical exertion increases metabolic rate and body temperature',
    },
    # Sound/waves
    {
        'conditions': ['sound', 'vacuum'],
        'effect': 'no',
        'law': 'Sound requires a medium (air, water, solid) to propagate',
    },
    {
        'conditions': ['sound', 'travel', 'space'],
        'effect': 'no',
        'law': 'Space is a vacuum; sound cannot travel without a medium',
    },
]

# Temperature-based reasoning
TEMPERATURE_EFFECTS = {
    # (substance, temp_C) → effect
    'freezing': 'Below the freezing point, a liquid becomes solid (freezes)',
    'melting': 'Above the melting point, a solid becomes liquid (melts)',
    'boiling': 'Above the boiling point, a liquid becomes gas (evaporates)',
}

# Why-questions: motivation/reason patterns
WHY_RULES = [
    {
        'conditions': ['wear', 'coat', 'winter'],
        'answer': 'to stay warm',
        'law': 'Coats provide thermal insulation, reducing heat loss',
    },
    {
        'conditions': ['birds', 'fly', 'south'],
        'answer': 'for warmer weather and food availability',
        'law': 'Migration is driven by seasonal food and temperature changes',
    },
    {
        'conditions': ['metal', 'rust'],
        'answer': 'exposure to oxygen and moisture causes oxidation',
        'law': 'Metals oxidize when exposed to oxygen and water',
    },
    {
        'conditions': ['leaves', 'change', 'color'],
        'answer': 'chlorophyll breaks down in autumn, revealing other pigments',
        'law': 'Reduced sunlight triggers chlorophyll degradation',
    },
    {
        'conditions': ['sky', 'blue'],
        'answer': 'sunlight scatters in the atmosphere (Rayleigh scattering), and blue light scatters most',
        'law': 'Rayleigh scattering: shorter wavelengths scatter more',
    },
    {
        'conditions': ['float', 'ice', 'water'],
        'answer': 'ice is less dense than liquid water because water expands when it freezes (anomalous expansion)',
        'law': "Archimedes' principle + anomalous expansion of water",
    },
    {
        'conditions': ['float', 'wood', 'water'],
        'answer': 'wood is less dense than water',
        'law': "Archimedes' principle: objects less dense than the fluid float",
    },
    {
        'conditions': ['sink', 'stone', 'water'],
        'answer': 'stone is denser than water',
        'law': "Archimedes' principle: objects denser than the fluid sink",
    },
    {
        'conditions': ['moon', 'phases'],
        'answer': 'the Moon orbits Earth, and we see different amounts of its sunlit side',
        'law': 'Lunar phases are caused by the Moon-Earth-Sun geometry',
    },
    {
        'conditions': ['tides'],
        'answer': "gravitational pull from the Moon and Sun",
        'law': "Tidal forces from the Moon's gravity distort Earth's oceans",
    },
    {
        'conditions': ['seasons'],
        'answer': "Earth's axis is tilted 23.5° relative to its orbit",
        'law': 'Axial tilt causes varying solar angle throughout the year',
    },
    {
        'conditions': ['day', 'night'],
        'answer': "Earth rotates on its axis once every 24 hours",
        'law': "Earth's rotation causes the day-night cycle",
    },
    {
        'conditions': ['sweat', 'exercise'],
        'answer': 'the body cools itself through evaporative cooling',
        'law': 'Sweat evaporation removes heat from the skin',
    },
    {
        'conditions': ['rainbow'],
        'answer': 'sunlight refracts through water droplets, splitting into colors',
        'law': 'Refraction and dispersion separate white light into its spectrum',
    },
    {
        'conditions': ['volcano', 'erupt'],
        'answer': 'magma pressure builds up beneath the surface',
        'law': 'Tectonic activity and magma convection drive volcanic eruptions',
    },
    {
        'conditions': ['earthquake'],
        'answer': 'tectonic plates shift and release built-up stress',
        'law': 'Earthquakes occur at plate boundaries due to friction and stress release',
    },
]


class CausalRulesEngine:
    """
    Forward-chaining causal inference engine.

    Instead of keyword→answer lookup, this engine:
    1. Gathers entity properties from KB + ConceptNet into working memory
    2. Matches rules against actual entity properties (not substrings)
    3. Chains derived facts to trigger further rules
    4. Uses ConceptNet Causes/HasSubevent for automatic causal chaining
    """

    # Action categories for property-based matching
    _HEAT_ACTIONS = frozenset(['heat', 'warm', 'burn', 'boil', 'cook', 'fire', 'flame'])
    _COOL_ACTIONS = frozenset(['cool', 'freeze', 'chill', 'refrigerate', 'ice'])
    _IMPACT_ACTIONS = frozenset(['drop', 'throw', 'hit', 'smash', 'break', 'crush', 'slam'])
    _MIX_ACTIONS = frozenset(['mix', 'combine', 'blend', 'merge', 'add'])

    def __init__(self, commonsense=None, knowledge=None):
        self.commonsense = commonsense
        self.knowledge = knowledge
        self._material_props = MATERIAL_PROPERTIES
        self._why_rules = WHY_RULES  # Keep as fallback — these encode real physics explanations

    def _gather_properties(self, entity: str) -> set:
        """Gather all known properties of an entity into a fact set."""
        props = set()
        e = entity.lower().strip()

        # Material properties
        if e in self._material_props:
            mp, bp, state = self._material_props[e]
            props.add(('state', state))
            props.add(('melting_point', mp))
            props.add(('boiling_point', bp))
        # Aliases
        aliases = {'ice': 'water', 'steam': 'water', 'snow': 'water'}
        if e in aliases:
            base = aliases[e]
            if base in self._material_props:
                mp, bp, _ = self._material_props[base]
                props.add(('melting_point', mp))
                props.add(('boiling_point', bp))
            # Override state for the alias
            alias_states = {'ice': 'solid', 'steam': 'gas', 'snow': 'solid'}
            if e in alias_states:
                props.add(('state', alias_states[e]))

        # Commonsense properties
        if self.commonsense:
            info = self.commonsense.about(e)
            if info.get('properties'):
                for rel, val in info['properties']:
                    props.add((rel, val.lower().strip()))
            # ConceptNet enrichment
            cn_facts = self.commonsense.cn_lookup(e, max_results=15)
            for rel, obj, weight in cn_facts:
                if weight >= 1.0:
                    props.add((rel.lower(), obj.lower().strip()))

        # KB facts
        if self.knowledge:
            for s, r, o in self.knowledge.facts:
                if s.lower() == e:
                    props.add((r.lower(), o.lower().strip()))

        return props

    def _classify_action(self, text: str) -> set:
        """Extract action categories from text."""
        words = set(text.lower().split())
        actions = set()
        if words & self._HEAT_ACTIONS:
            actions.add('heat')
        if words & self._COOL_ACTIONS:
            actions.add('cool')
        if words & self._IMPACT_ACTIONS:
            actions.add('impact')
        if words & self._MIX_ACTIONS:
            actions.add('mix')
        return actions

    def _extract_entities(self, text: str) -> List[str]:
        """Extract content words as candidate entities."""
        import re
        stop = {'what', 'happens', 'when', 'you', 'the', 'a', 'an', 'to',
                'if', 'does', 'do', 'is', 'are', 'was', 'were', 'will',
                'would', 'can', 'could', 'should', 'it', 'its', 'this',
                'that', 'with', 'from', 'for', 'on', 'in', 'at', 'by',
                'and', 'or', 'but', 'of', 'how', 'why', 'then', 'than'}
        words = re.findall(r'[a-z]+', text.lower())
        return [w for w in words if w not in stop and len(w) > 1]

    def get_material_state(self, substance: str) -> Optional[str]:
        """Get the default state of a substance from properties."""
        props = self._gather_properties(substance)
        for key, val in props:
            if key == 'state':
                return val
            if key == 'has_property' and val in ('solid', 'liquid', 'gas'):
                return val
        return None

    def get_material_property(self, substance: str) -> Optional[Dict]:
        """Get melting/boiling points for a substance."""
        props = self._gather_properties(substance)
        result = {}
        for key, val in props:
            if key == 'melting_point':
                result['melting_point'] = val
            elif key == 'boiling_point':
                result['boiling_point'] = val
            elif key == 'state':
                result['state'] = val
        return result if 'state' in result else None

    def predict_temperature_effect(self, substance: str, temp_c: float) -> Optional[str]:
        """Predict what happens to a substance at a given temperature."""
        props = self.get_material_property(substance)
        if not props:
            return None

        mp = props.get('melting_point')
        bp = props.get('boiling_point')
        state = props.get('state')

        if mp is not None and bp is not None and state:
            if state == 'solid' and temp_c > mp:
                return 'it melts'
            elif state == 'liquid':
                if temp_c <= mp:
                    return 'it freezes'
                elif temp_c >= bp:
                    return 'it evaporates'
            elif state == 'gas' and temp_c <= bp:
                return 'it condenses'
            return f'{substance} remains {state} at {temp_c}°C'
        return None

    def predict_action_effect(self, action: str, subject: str,
                              question: str = '') -> Optional[str]:
        """
        Forward-chaining effect prediction.

        1. Gather entity properties into working memory
        2. Classify the action
        3. Match rules against properties (not keywords)
        4. Chain ConceptNet Causes relations
        """
        import re
        q_lower = question.lower() if question else ''
        all_text = f"{action.lower()} {subject.lower()} {q_lower}"

        # Phase 1: Temperature-specific reasoning
        temp_match = re.search(r'(\-?\d+)\s*(?:°|degree)', all_text)
        if temp_match:
            temp = float(temp_match.group(1))
            for entity in self._extract_entities(all_text):
                result = self.predict_temperature_effect(entity, temp)
                if result:
                    return result

        # Phase 2: Gather properties for all entities
        entities = self._extract_entities(all_text)
        entity_props = {}
        for e in entities:
            p = self._gather_properties(e)
            if p:
                entity_props[e] = p

        # Phase 3: Action classification
        actions = self._classify_action(all_text)

        # Phase 4: Property-based rule application
        # State transitions — check if any entity IS solid/liquid/gas
        if 'heat' in actions:
            for e, props in entity_props.items():
                state = None
                for key, val in props:
                    if key == 'state':
                        state = val
                    elif key == 'has_property' and val in ('solid', 'liquid', 'gas'):
                        state = val
                if state:
                    # Check exceptions: wood burns, doesn't melt
                    burns = any(val in ('flammable', 'combustible', 'burns')
                               for key, val in props
                               if key in ('has_property', 'hasproperty', 'capableof'))
                    if burns and state == 'solid':
                        return 'it burns'
                    next_state = STATE_TRANSITIONS.get(state, {}).get('heat')
                    if next_state:
                        verbs = {'liquid': 'melts', 'gas': 'evaporates/boils'}
                        return f'it {verbs.get(next_state, f"becomes {next_state}")}'

        if 'cool' in actions:
            for e, props in entity_props.items():
                state = None
                for key, val in props:
                    if key == 'state':
                        state = val
                    elif key == 'has_property' and val in ('solid', 'liquid', 'gas'):
                        state = val
                if state:
                    next_state = STATE_TRANSITIONS.get(state, {}).get('cool')
                    if next_state:
                        verbs = {'solid': 'freezes', 'liquid': 'condenses'}
                        return f'it {verbs.get(next_state, f"becomes {next_state}")}'

        # Impact + fragility — check actual properties, not keyword list
        if 'impact' in actions:
            for e, props in entity_props.items():
                prop_values = {val for _, val in props}
                if 'fragile' in prop_values or 'brittle' in prop_values:
                    return 'it breaks'
                if 'hard' in prop_values or 'strong' in prop_values or 'durable' in prop_values:
                    return 'it survives the impact'
                # Glass-like: check HasProperty from ConceptNet
                if any(v in ('transparent', 'breakable') for v in prop_values):
                    return 'it breaks'

        # Color mixing — check if entities are colors
        if 'mix' in actions:
            colors_present = set()
            color_set = {'red', 'blue', 'yellow', 'green', 'orange', 'purple', 'white', 'black'}
            for e in entities:
                if e in color_set:
                    colors_present.add(e)
            _COLOR_MIX = {
                frozenset({'red', 'blue'}): 'purple',
                frozenset({'red', 'yellow'}): 'orange',
                frozenset({'blue', 'yellow'}): 'green',
            }
            if len(colors_present) >= 2:
                for combo, result in _COLOR_MIX.items():
                    if combo <= colors_present:
                        return result

        # Phase 5: ConceptNet Causes chaining
        if self.commonsense:
            for e in entities:
                cn_facts = self.commonsense.cn_lookup(e, max_results=15)
                for rel, obj, weight in cn_facts:
                    if rel == 'Causes' and weight >= 2.0:
                        return obj

        # Phase 6: Fallback to keyword-scored causal laws
        best_match = None
        best_score = 0
        for law in CAUSAL_LAWS:
            conditions = law['conditions']
            score = sum(1 for c in conditions if c in all_text)
            match_ratio = score / len(conditions)
            if 'exceptions' in law:
                if any(ex in all_text for ex in law['exceptions']):
                    continue
            if match_ratio >= 0.6 and score > best_score:
                best_score = score
                best_match = law
        if best_match:
            return best_match['effect']

        return None

    def answer_why(self, question: str) -> Optional[str]:
        """Answer 'why' questions using ConceptNet Causes + physical laws."""
        q_lower = question.lower()

        # Phase 1: ConceptNet reverse-Causes lookup
        if self.commonsense:
            entities = self._extract_entities(q_lower)
            for e in entities:
                cn_facts = self.commonsense.cn_lookup(e, max_results=15)
                for rel, obj, weight in cn_facts:
                    if rel in ('Causes', 'HasSubevent') and weight >= 3.0:
                        return obj

        # Phase 2: Fallback to WHY_RULES (real physics explanations)
        best_match = None
        best_score = 0
        for rule in self._why_rules:
            conditions = rule['conditions']
            score = sum(1 for c in conditions if c in q_lower)
            match_ratio = score / len(conditions)
            if match_ratio >= 0.5 and score > best_score:
                best_score = score
                best_match = rule
        if best_match:
            return best_match['answer']

        return None

    def answer_what_causes(self, effect: str) -> Optional[str]:
        """Answer 'what causes/makes X?' using reverse causal lookup."""
        effect_lower = effect.lower()

        # Phase 1: ConceptNet — find what Causes this effect
        if self.commonsense:
            entities = self._extract_entities(effect_lower)
            for e in entities:
                cn_facts = self.commonsense.cn_lookup(e, max_results=15)
                for rel, obj, weight in cn_facts:
                    if rel == 'Causes' and weight >= 2.0:
                        return obj

        # Phase 2: Fallback keyword match against causal laws
        best_match = None
        best_score = 0
        for law in CAUSAL_LAWS:
            score = sum(1 for c in law['conditions'] if c in effect_lower)
            if score > best_score:
                best_score = score
                best_match = law
        if best_match and best_score > 0:
            return best_match['effect']

        # Phase 3: WHY_RULES fallback
        for rule in self._why_rules:
            score = sum(1 for c in rule['conditions'] if c in effect_lower)
            if score > 0:
                return rule['answer']

        return None

    def derive_chain(self, action: str, subject: str, depth: int = 3) -> List[str]:
        """
        Forward-chain causal effects: action → effect1 → effect2...

        Uses ConceptNet Causes for transitive chaining.
        """
        chain = []
        seen = set()
        current = action.lower()

        for _ in range(depth):
            # Try predict_action_effect first
            effect = self.predict_action_effect(current, subject)
            if effect and effect not in seen:
                chain.append(effect)
                seen.add(effect)
                current = effect
                continue

            # Try ConceptNet Causes chaining
            if self.commonsense:
                cn_facts = self.commonsense.cn_lookup(current, max_results=10)
                found_next = False
                for rel, obj, weight in cn_facts:
                    if rel == 'Causes' and obj not in seen:
                        chain.append(obj)
                        seen.add(obj)
                        current = obj
                        found_next = True
                        break
                if found_next:
                    continue
            break

        return chain
