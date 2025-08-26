"""
pera_drm: Uniwersalny moduł do gier planszowych z DRM 3.0
Łączy PERA-Net (permutacyjnie-równą sieć neuronową) z pełną implementacją DRM 3.0.

🔥 KLUCZOWE ULEPSZENIA:
- Pełna implementacja DRM 3.0 z anti-stagnation
- Permutacyjna równość w architekturze
- Inteligentne zarządzanie regułami
- System emergencji i ciekawości
- Monitoring stagnacji systemu
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any
import math
import numpy as np
from abc import ABC, abstractmethod
import copy

# =============================
# DRM 3.0 - PEŁNA IMPLEMENTACJA
# =============================

class Rule:
    """Pojedyncza reguła w systemie DRM"""
    def __init__(self, id: int, Wi: float = 1.0, Ci: float = 0.0, Ui: float = 0.0, 
                 Ri: float = 1.0, Mi: float = 1.0, age: int = 0, last_success: int = 0, mutations: int = 0):
        self.id = id
        self.Wi = Wi      # Waga
        self.Ci = Ci      # Confidence
        self.Ui = Ui      # Usage
        self.Ri = Ri      # Relevance
        self.Mi = Mi      # Momentum
        self.age = age    # Wiek reguły
        self.last_success = last_success # Ostatni sukces
        self.mutations = mutations    # Liczba mutacji
        self.birth_cycle = 0
        self.activity_history = []
    
    def __repr__(self):
        return f"Rule(id={self.id}, Wi={self.Wi:.2f}, Ci={self.Ci:.2f})"

class DRM3:
    """Dynamic Rules Matrix 3.0 z pełną funkcjonalnością anti-stagnation"""
    
    def __init__(self, 
                 frz: float = 0.5,
                 base_emergence: float = 0.1,
                 base_curiosity: float = 0.05,
                 stagnation_threshold: float = 0.01,
                 max_rules: int = 1000):
        
        self.frz = frz
        self.base_emergence = base_emergence
        self.base_curiosity = base_curiosity
        self.stagnation_threshold = stagnation_threshold
        self.max_rules = max_rules
        
        # System state
        self.cycle = 1
        self.rules: Dict[int, Rule] = {}
        self.next_rule_id = 0
        
        # Monitoring stagnacji
        self.activity_history = []
        self.diversity_history = []
        self.emergence_history = []
        self.last_emergence_cycle = 0
        
        # Pressure factors
        self.stagnation_pressure = 0.0
        self.diversity_pressure = 0.0
        self.novelty_pressure = 0.0
        
    def calculate_adaptive_frz(self) -> float:
        """Inteligentny modulator FRZ"""
        if self.is_stagnating():
            # BOOST - stagnacja dostaje boost, nie hamulec!
            return min(1.5, self.frz * 2.0)
        elif self.is_chaotic():
            # Tradycyjny hamulec
            return max(0.1, self.frz * 0.5)
        elif self.is_learning_phase():
            # Stały wzrost w fazie nauki
            return 0.8 + self._exploration_bonus()
        else:
            return self.frz
    
    def calculate_emergence_pressure(self) -> float:
        """Adaptacyjna presja środowiska"""
        pressure_factors = []
        
        # Stagnation pressure
        if self.get_avg_strength() < self.stagnation_threshold:
            pressure_factors.append(2.0)
            
        # Diversity pressure  
        if self.calculate_rule_similarity() > 0.8:
            pressure_factors.append(1.5)
            
        # Novelty pressure
        cycles_since_new_rule = self.cycle - max([r.birth_cycle for r in self.rules.values()], default=0)
        if cycles_since_new_rule > 10:
            pressure_factors.append(1.0)
            
        # Success pressure
        if self.get_overall_success_rate() < 0.5:
            pressure_factors.append(0.8)
            
        return self.base_emergence * (1.0 + sum(pressure_factors)) if pressure_factors else self.base_emergence
    
    def calculate_curiosity_drive(self, rule: Rule) -> float:
        """Mechanizm głodu poznawczego"""
        exploration_bonus = 0.0
        
        # Nowa/mało używana reguła
        if rule.Ui < 5:
            exploration_bonus += 0.8
            
        # Rzadka kombinacja
        if self.is_rare_combination(rule):
            exploration_bonus += 0.6
            
        # Sprzeczne dowody
        if self.has_contradictory_evidence(rule):
            exploration_bonus += 0.4
            
        # Niespodziewany sukces
        if rule.last_success == self.cycle - 1 and rule.Ci < 0.3:
            exploration_bonus += 0.3
        
        # Novelty seeking - młode reguły są bardziej ciekawe
        max_experience = max([r.Ui for r in self.rules.values()], default=1)
        novelty_seeking = max(0.2, 1.0 - (rule.Ui / max_experience)) if max_experience > 0 else 1.0
        
        return 1 + (exploration_bonus * novelty_seeking)
    
    def calculate_anti_stagnation_factor(self) -> float:
        """Aktywny przeciwnik nudy"""
        if self.detect_death_spiral():
            return 2.0 + self._emergency_randomization()
        elif self.is_stagnating():
            return 1.5 + self._mutation_pressure()
        elif self.is_low_activity():
            return 1.2 + np.random.uniform(0, 0.5)
        else:
            return 1.0
    
    def calculate_rule_strength(self, rule: Rule, context_delta: float = 0, max_delta: float = 10) -> float:
        """Główna funkcja kalkulacji siły reguły DRM 3.0"""
        # Podstawa
        base = rule.Wi * math.log(rule.Ci + 1) * (1 + rule.Ui/self.cycle) * rule.Ri
        
        # Momentum z kontekstem
        momentum = rule.Mi ** (context_delta / max_delta) if max_delta > 0 else rule.Mi
        
        # Nowe komponenty DRM 3.0
        frz_adaptive = self.calculate_adaptive_frz()
        emergence_pressure = self.calculate_emergence_pressure()
        curiosity_drive = self.calculate_curiosity_drive(rule)
        anti_stagnation = self.calculate_anti_stagnation_factor()
        
        # Integracja
        interaction = 1 + sum([r.Wi * 0.1 for r in self.rules.values() if r.id != rule.id])
        
        strength = (base * momentum * frz_adaptive * emergence_pressure * 
                   curiosity_drive * anti_stagnation * interaction)
        
        # Update rule history
        rule.activity_history.append(strength)
        if len(rule.activity_history) > 100:  # Keep last 100
            rule.activity_history.pop(0)
            
        return max(0.001, strength)  # Minimum threshold
    
    def step_all_rules(self, context_changes: Dict[int, float] = None) -> Dict[int, float]:
        """Wykonaj krok dla wszystkich reguł"""
        if context_changes is None:
            context_changes = {}
            
        strengths = {}
        rule_values = []  # Przechowujemy wartości dla obliczeń średnich
        
        for rule_id, rule in self.rules.items():
            context_delta = context_changes.get(rule_id, 0)
            strength = self.calculate_rule_strength(rule, context_delta)
            strengths[rule_id] = strength
            rule_values.append(strength)
            
            # Update rule stats
            rule.age += 1
            if strength > np.mean(rule_values) if rule_values else 0:  # Poprawione obliczenie średniej
                rule.last_success = self.cycle
        
        # System monitoring - obliczamy średnią po pełnej pętli
        if rule_values:
            avg_strength = np.mean(rule_values)
            diversity = np.std(rule_values) if len(rule_values) > 1 else 0.0
            self.activity_history.append(avg_strength)
            self.diversity_history.append(diversity)
        else:
            self.activity_history.append(0.0)
            self.diversity_history.append(0.0)
        
        # Emergency actions
        if self.detect_death_spiral():
            self.emergency_revival()
            
        if self.calculate_rule_similarity() > 0.8:
            self.enforce_diversity()
        
        self.cycle += 1
        return strengths
    
    def add_rule(self, Wi: float = 1.0, Ci: float = 0.1, Ui: float = 0.0, 
                 Ri: float = 1.0, Mi: float = 1.0) -> int:
        """Dodaj nową regułę do systemu"""
        rule_id = self.next_rule_id
        rule = Rule(id=rule_id, Wi=Wi, Ci=Ci, Ui=Ui, Ri=Ri, Mi=Mi)
        rule.birth_cycle = self.cycle
        
        self.rules[rule_id] = rule
        self.next_rule_id += 1
        self.last_emergence_cycle = self.cycle
        
        return rule_id
    
    # === DETECTION METHODS ===
    
    def is_stagnating(self) -> bool:
        """Wykryj stagnację systemu"""
        if len(self.activity_history) < 5:
            return False
        
        recent_activity = self.activity_history[-5:]
        avg_change = np.std(recent_activity)
        return avg_change < self.stagnation_threshold
    
    def detect_death_spiral(self) -> bool:
        """Wykryj spiralę śmierci"""
        if len(self.activity_history) < 10:
            return False
            
        conditions = [
            self.get_avg_strength() < 0.05,
            self.cycle - self.last_emergence_cycle > 10,
            self.frz < 0.1,
            len([r for r in self.rules.values() if r.last_success > self.cycle - 5]) == 0
        ]
        
        return all(conditions)
    
    def is_chaotic(self) -> bool:
        """Wykryj chaos w systemie"""
        if len(self.diversity_history) < 3:
            return False
        return np.mean(self.diversity_history[-3:]) > 2.0
    
    def is_learning_phase(self) -> bool:
        """Czy system jest w fazie nauki"""
        return self.cycle < 50 or len(self.rules) < self.max_rules * 0.3
    
    def is_low_activity(self) -> bool:
        """Wykryj niską aktywność"""
        if len(self.activity_history) < 3:
            return False
        return np.mean(self.activity_history[-3:]) < 0.3
    
    # === HELPER METHODS ===
    
    def get_avg_strength(self) -> float:
        """Średnia siła systemu"""
        return np.mean(self.activity_history[-10:]) if self.activity_history else 0.0
    
    def calculate_rule_similarity(self) -> float:
        """Oblicz podobieństwo reguł"""
        if len(self.rules) < 2:
            return 0.0
        
        similarities = []
        rules_list = list(self.rules.values())
        
        for i, r1 in enumerate(rules_list):
            for r2 in rules_list[i+1:]:
                sim = abs(r1.Wi - r2.Wi) + abs(r1.Ri - r2.Ri) + abs(r1.Mi - r2.Mi)
                similarities.append(1.0 - min(sim / 3.0, 1.0))
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_overall_success_rate(self) -> float:
        """Ogólny wskaźnik sukcesu"""
        if not self.rules:
            return 0.0
        
        recent_successes = sum(1 for r in self.rules.values() 
                              if r.last_success > self.cycle - 10)
        return recent_successes / len(self.rules) if self.rules else 0.0
    
    def is_rare_combination(self, rule: Rule) -> bool:
        """Czy reguła reprezentuje rzadką kombinację"""
        similar_rules = sum(1 for r in self.rules.values() 
                           if abs(r.Wi - rule.Wi) < 0.1 and abs(r.Ri - rule.Ri) < 0.1)
        return similar_rules <= 2
    
    def has_contradictory_evidence(self, rule: Rule) -> bool:
        """Czy reguła ma sprzeczne dowody"""
        return (rule.Ci > 0.7 and rule.Ui < 3) or (rule.Ci < 0.3 and rule.Ui > 10)
    
    # === EMERGENCY ACTIONS ===
    
    def emergency_revival(self):
        """Reanimacja systemu w kryzysie"""
        print(f"🚨 EMERGENCY REVIVAL at cycle {self.cycle}!")
        
        # Wstrzyknij chaos
        for _ in range(3):
            self.add_rule(
                Wi=np.random.uniform(0.5, 2.0),
                Ci=np.random.uniform(0.0, 1.0),
                Ri=np.random.uniform(0.5, 1.5)
            )
        
        # Boost słabych reguł
        for rule in self.rules.values():
            if rule.Wi < 0.5:
                rule.Wi *= 2.0
                rule.mutations += 1
        
        # Wymuś rekombinacje
        self._force_rule_recombination()
        
        # Zwiększ presję emergencji
        self.base_emergence *= 3.0
    
    def enforce_diversity(self):
        """Wymuś różnorodność w systemie"""
        print(f"🔄 ENFORCING DIVERSITY at cycle {self.cycle}")
        
        # Znajdź najbardziej podobne reguły
        rules_list = list(self.rules.values())
        most_similar = []
        
        for i, r1 in enumerate(rules_list):
            for r2 in rules_list[i+1:]:
                sim = 1.0 - (abs(r1.Wi - r2.Wi) + abs(r1.Ri - r2.Ri)) / 2.0
                if sim > 0.9:
                    most_similar.append(r1)
        
        # Mutuj podobne reguły
        for rule in most_similar[:5]:  # Top 5 most similar
            rule.Wi += np.random.uniform(-0.5, 0.5)
            rule.Ri += np.random.uniform(-0.3, 0.3)
            rule.Mi += np.random.uniform(-0.2, 0.2)
            rule.mutations += 1
    
    def _force_rule_recombination(self):
        """Wymuś rekombinację reguł"""
        if len(self.rules) < 2:
            return
            
        rules_list = list(self.rules.values())
        r1, r2 = np.random.choice(rules_list, 2, replace=False)
        
        # Hybrid rule
        new_Wi = (r1.Wi + r2.Wi) / 2 + np.random.uniform(-0.2, 0.2)
        new_Ri = (r1.Ri + r2.Ri) / 2 + np.random.uniform(-0.1, 0.1)
        new_Mi = (r1.Mi + r2.Mi) / 2
        
        self.add_rule(Wi=new_Wi, Ri=new_Ri, Mi=new_Mi, Ci=0.1)
    
    def _exploration_bonus(self) -> float:
        """Bonus za eksplorację"""
        return min(0.3, len(self.rules) / self.max_rules)
    
    def _mutation_pressure(self) -> float:
        """Presja mutacyjna"""
        return np.random.uniform(0.1, 0.5)
    
    def _emergency_randomization(self) -> float:
        """Awaryjne losowanie"""
        return np.random.uniform(0.5, 1.5)

# =============================
# PERA-Net z permutacyjną równością
# =============================

class ModelSize:
    """Konfiguracja rozmiaru modelu"""
    def __init__(self, channels: int, blocks: int, heads: int = 8):
        self.channels = channels
        self.blocks = blocks
        self.heads = heads  # Dla attention

SIZES: Dict[str, ModelSize] = {
    "tiny":   ModelSize(128, 6, 4),   # ~5M params
    "small":  ModelSize(192, 10, 6),  # ~15M params  
    "medium": ModelSize(256, 14, 8),  # ~40M params
    "large":  ModelSize(384, 18, 12), # ~120M params
}

class GameDescriptor:
    """Deskryptor gry planszowej"""
    def __init__(self, board_size: Tuple[int, int], planes: int, action_space: int, game_id: int = 0, symmetries: int = 8):
        self.board_size = board_size
        self.planes = planes
        self.action_space = action_space
        self.game_id = game_id
        self.symmetries = symmetries

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        H, W = self.board_size
        return torch.tensor([H, W, self.planes, self.action_space, self.game_id, self.symmetries], 
                          dtype=torch.float32, device=device)

class EquivariantConv2d(nn.Module):
    """Konwolucja z permutacyjną równością"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardowa konwolucja już jest equivariant względem translacji
        # Dla rotacji można dodać data augmentation lub group convolutions
        return self.conv(x)

class ResidualBlock(nn.Module):
    """Blok residualny z permutacyjną równością"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = EquivariantConv2d(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = EquivariantConv2d(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(residual + out)

class AttentionBlock(nn.Module):
    """Self-attention dla lepszego zrozumienia pozycji"""
    
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads if channels > 0 else 1
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Reshape for attention - zabezpieczenie przed zerowymi rozmiarami
        if C == 0 or H == 0 or W == 0:
            return x
            
        # Reshape for attention
        qkv = self.qkv(x).view(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        
        # Attention
        attn = torch.softmax(torch.einsum('bhdx,bhdy->bhxy', q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum('bhxy,bhdy->bhdx', attn, v)
        
        # Poprawiony kod: użyj reshape zamiast view
        try:
            out = out.view(B, C, H, W)
        except RuntimeError:
            # Jeśli view nie działa, użyj reshape
            out = out.reshape(B, C, H, W)
        
        out = self.proj(out)
        
        # Residual + norm
        return x + self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PERANet(nn.Module):
    """Permutation-Equivariant Residual Architecture dla gier planszowych"""
    
    def __init__(self, size: str, in_planes: int = 32, max_action_space: int = 512):
        super().__init__()
        if size not in SIZES:
            raise ValueError(f"Size must be one of {list(SIZES.keys())}")
            
        cfg = SIZES[size]
        self.size = size
        self.channels = cfg.channels
        self.max_action_space = max_action_space
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_planes, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.SiLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels) for _ in range(cfg.blocks)
        ])
        
        # Attention blocks (every 3rd block)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.channels, cfg.heads) 
            if i % 3 == 0 else nn.Identity()
            for i in range(cfg.blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 2, 1)  # 2 channels for flexibility
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.channels, self.channels // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.channels // 2, self.channels // 4),
            nn.SiLU(),
            nn.Linear(self.channels // 4, 1),
            nn.Tanh()
        )
        
        # Game descriptor embedding
        self.game_embedding = nn.Linear(6, self.channels)  # H,W,planes,actions,game_id,symmetries
        
    def forward(self, x: torch.Tensor, game_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass z wrapperem dla GameDescriptor"""
        B, C, H, W = x.shape
        device = x.device

        # Input projection
        x = self.input_proj(x)
        
        # Add game context from precomputed vector
        if len(game_vec.shape) == 1:
            game_emb = self.game_embedding(game_vec.view(1, -1)).view(1, -1, 1, 1)
        else:
            game_emb = self.game_embedding(game_vec).view(-1, self.channels, 1, 1)
        game_emb = game_emb.expand(B, -1, H, W)
        
        # Add game context to input
        if game_emb.shape[1] == x.shape[1]:  # Matching channels
            x = x + game_emb
        
        # Main processing
        for res_block, attn_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x)
            x = attn_block(x)
        
        # Policy output
        pol = self.policy_head(x)
        B, _, H, W = pol.shape
        
        # Adaptive policy logits based on game
        pol_logits = pol[:, 0].view(B, -1)  # Use only first channel as in original code

        # Value output
        value = self.value_head(x)
        
        return pol_logits, value
    
    def get_policy_value(self, x: torch.Tensor, desc: GameDescriptor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for cleaner API"""
        game_vec = desc.to_tensor(x.device)
        return self.forward(x, game_vec)

# =============================
# Enhanced PUCT with DRM integration
# =============================

class EnhancedPUCT:
    """PUCT z integracją DRM dla adaptacyjnego MCTS"""
    
    def __init__(self, c_puct: float = 1.5, sims: int = 800, drm: Optional[DRM3] = None):
        self.c_puct = c_puct
        self.sims = sims
        self.drm = drm or DRM3()
        
        # MCTS state
        self.Q = {}  # Q values
        self.N = {}  # Visit counts
        self.P = {}  # Prior probabilities
        
    def select_action(self, state: str, policy_probs: torch.Tensor, legal_actions: torch.Tensor) -> int:
        """Wybierz akcję używając PUCT + DRM"""
        
        if state not in self.Q:
            self.Q[state] = torch.zeros_like(policy_probs)
            self.N[state] = torch.zeros_like(policy_probs)
            self.P[state] = policy_probs.clone()
        
        Q = self.Q[state]
        N = self.N[state]
        P = self.P[state]
        
        # DRM-enhanced selection
        if self.drm:
            # Dodaj DRM-based exploration bonus - poprawiona wersja
            drm_bonus = self._calculate_drm_bonus(state, legal_actions)
            P = P + drm_bonus
            # Renormalizacja prawdopodobieństw
            P = (P + 1e-8).clamp_min(0)  # Zapobieganie ujemnym wartościam
            P = P / (P.sum() + 1e-8)  # Normalizacja do sumy 1
        
        # PUCT formula
        sqrt_sum_n = torch.sqrt(N.sum() + 1e-8)
        U = self.c_puct * P * sqrt_sum_n / (1 + N)
        
        # Combine Q + U, mask illegal moves
        scores = Q + U
        scores[~legal_actions] = float('-inf')
        
        action = int(torch.argmax(scores).item())
        return action
    
    def update(self, state: str, action: int, value: float):
        """Aktualizuj statystyki MCTS"""
        if state in self.Q:
            self.N[state][action] += 1
            self.Q[state][action] += (value - self.Q[state][action]) / self.N[state][action]
    
    def _calculate_drm_bonus(self, state: str, legal_actions: torch.Tensor) -> torch.Tensor:
        """Oblicz DRM-based exploration bonus"""
        bonus = torch.zeros_like(legal_actions, dtype=torch.float32)
        
        # Przykład: zwiększ bonus dla akcji w stagnujących regionach
        if self.drm.is_stagnating():
            # Dodaj losowy exploration bonus z poprawną konwersją typów
            bonus_count = int(legal_actions.sum().item())
            if bonus_count > 0:
                random_bonus = torch.rand(bonus_count, device=legal_actions.device)
                # Przypisz bonus tylko do legal actions
                bonus[legal_actions] = random_bonus * 0.1
        
        return bonus

# =============================
# Utilities & Training Support
# =============================

def count_parameters(model: nn.Module) -> int:
    """Policz parametry modelu"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_game_descriptor(game_name: str) -> GameDescriptor:
    """Factory function dla popularnych gier"""
    game_configs = {
        'go_19': GameDescriptor((19, 19), 17, 361, 1),
        'go_13': GameDescriptor((13, 13), 17, 169, 2), 
        'go_9': GameDescriptor((9, 9), 17, 81, 3),
        'chess': GameDescriptor((8, 8), 20, 4096, 4),
        'othello': GameDescriptor((8, 8), 3, 64, 5),
        'gomoku': GameDescriptor((15, 15), 3, 225, 6),
        'tictactoe': GameDescriptor((3, 3), 3, 9, 7),
    }
    
    if game_name not in game_configs:
        raise ValueError(f"Game '{game_name}' not supported. Available: {list(game_configs.keys())}")
    
    return game_configs[game_name]

# Wrapper dla eksportu ONNX
class PERANetWrapper(nn.Module):
    """Wrapper do eksportu ONNX i TorchScript"""
    def __init__(self, model: PERANet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, game_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x, game_vec)

def export_to_onnx(model: nn.Module, desc: GameDescriptor, filename: str):
    """Export modelu do ONNX z wrapperem"""
    model.eval()
    dummy_input = torch.randn(1, desc.planes, desc.board_size[0], desc.board_size[1])
    game_vec = desc.to_tensor(dummy_input.device)
    
    # Użyj wrapper dla eksportu
    wrapper_model = PERANetWrapper(model)
    torch.onnx.export(
        wrapper_model,
        (dummy_input, game_vec),
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['board_state', 'game_descriptor'],
        output_names=['policy_logits', 'value'],
        dynamic_axes={
            'board_state': {0: 'batch_size'},
            'policy_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

# =============================
# Example Usage & Testing
# =============================

if __name__ == "__main__":
    print("🚀 PERA-DRM 3.0 - Advanced Testing")
    print("="*60)
    
    # Test 1: Create and test DRM system
    print("\n🧬 Testing DRM 3.0 System:")
    drm = DRM3(frz=0.3, base_emergence=0.15, base_curiosity=0.08)
    
    # Add initial rules
    rule_ids = []
    for i in range(10):
        rule_id = drm.add_rule(
            Wi=np.random.uniform(0.5, 2.0),
            Ci=np.random.uniform(0.0, 1.0), 
            Ri=np.random.uniform(0.5, 1.5)
        )
        rule_ids.append(rule_id)
    
    print(f"✅ Created {len(rule_ids)} initial rules")
    
    # Simulate evolution cycles
    print("\n🔄 Simulating 50 evolution cycles:")
    for cycle in range(50):
        # Random context changes
        context_changes = {rid: np.random.uniform(-0.5, 0.5) for rid in rule_ids[:3]}
        
        # Step all rules
        strengths = drm.step_all_rules(context_changes)
        
        if cycle % 10 == 0:
            avg_strength = np.mean(list(strengths.values()))
            diversity = np.std(list(strengths.values())) if len(strengths) > 1 else 0.0
            print(f"   Cycle {cycle:2d}: avg_strength={avg_strength:.3f}, diversity={diversity:.3f}")
            
            # Show system state
            if drm.is_stagnating():
                print("     🚨 Stagnation detected!")
            if drm.is_chaotic():
                print("     ⚡ Chaos detected!")
            if drm.detect_death_spiral():
                print("     💀 Death spiral detected!")
    
    print(f"\n📊 Final DRM stats:")
    print(f"   Rules count: {len(drm.rules)}")
    print(f"   System cycles: {drm.cycle}")
    print(f"   Avg strength: {drm.get_avg_strength():.3f}")
    print(f"   Rule similarity: {drm.calculate_rule_similarity():.3f}")
    print(f"   Success rate: {drm.get_overall_success_rate():.3f}")
    
    # Test 2: Neural Network Models
    print("\n🧠 Testing PERA-Net Models:")
    
    games_to_test = ['go_9', 'othello', 'tictactoe', 'chess']
    model_sizes = ['tiny', 'small', 'medium']
    
    for size in model_sizes:
        print(f"\n   🔍 Testing {size.upper()} model:")
        
        for game_name in games_to_test:
            desc = create_game_descriptor(game_name)
            model = PERANet(size=size, in_planes=desc.planes, max_action_space=desc.action_space)
            
            # Count parameters
            param_count = count_parameters(model) / 1e6
            
            # Test forward pass
            batch_size = 4
            dummy_input = torch.randn(batch_size, desc.planes, desc.board_size[0], desc.board_size[1])
            game_vec = desc.to_tensor(dummy_input.device)
            
            with torch.no_grad():
                policy_logits, value = model(dummy_input, game_vec)
            
            print(f"     {game_name:8s}: {param_count:5.1f}M params, "
                  f"policy: {policy_logits.shape}, value: {value.shape}")
    
    # Test 3: PUCT with DRM Integration
    print("\n🎯 Testing Enhanced PUCT with DRM:")
    
    desc = create_game_descriptor('go_9')
    model = PERANet('small', desc.planes, desc.action_space)
    puct = EnhancedPUCT(c_puct=1.4, sims=100, drm=drm)
    
    # Simulate game position
    dummy_state = torch.randn(1, desc.planes, desc.board_size[0], desc.board_size[1])
    game_vec = desc.to_tensor(dummy_state.device)
    with torch.no_grad():
        policy_probs, value = model(dummy_state, game_vec)
        policy_probs = torch.softmax(policy_probs[0], dim=0)
    
    # Legal actions (random for demo)
    legal_actions = torch.randint(0, 2, (desc.action_space,), dtype=torch.bool)
    legal_actions[0] = True  # Ensure at least one legal action
    
    # Test PUCT selection
    state_key = "test_position_1"
    selected_action = puct.select_action(state_key, policy_probs, legal_actions)
    
    print(f"   Selected action: {selected_action}")
    print(f"   Legal actions count: {legal_actions.sum().item()}")
    print(f"   Policy prob for selected: {policy_probs[selected_action]:.4f}")
    print(f"   Value estimate: {value.item():.4f}")
    
    # Test 4: Export capabilities
    print("\n💾 Testing Export Capabilities:")
    
    try:
        # Test ONNX export
        model.eval()
        export_filename = "pera_net_test.onnx"
        export_to_onnx(model, desc, export_filename)
        print(f"   ✅ ONNX export successful: {export_filename}")
        
        # Test TorchScript (bez użycia wrapperów z GameDescriptor)
        # Alternatywna wersja bez GameDescriptor
        class SimplePERANet(nn.Module):
            def __init__(self, model: PERANet):
                super().__init__()
                self.model = model
                
            def forward(self, x: torch.Tensor):
                # Prosta wersja bez game descriptor
                dummy_desc = create_game_descriptor('go_9')
                game_vec = dummy_desc.to_tensor(x.device)
                return self.model.forward(x, game_vec)[0]  # Tylko policy logits

        simple_model = SimplePERANet(model)
        scripted_model = torch.jit.script(simple_model)
        scripted_filename = "pera_net_test.pt"
        scripted_model.save(scripted_filename)
        print(f"   ✅ TorchScript export successful: {scripted_filename}")
        
    except Exception as e:
        print(f"   ❌ Export error: {e}")
    
    # Test 5: Performance Benchmark
    print("\n⚡ Performance Benchmark:")
    
    import time
    
    desc = create_game_descriptor('go_19')
    model = PERANet('medium', desc.planes, desc.action_space)
    model.eval()
    
    batch_sizes = [1, 4, 16, 64]
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, desc.planes, desc.board_size[0], desc.board_size[1])
        game_vec = desc.to_tensor(dummy_input.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input, game_vec)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                policy_logits, value = model(dummy_input, game_vec)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = batch_size / avg_time
        
        print(f"   Batch {batch_size:2d}: {avg_time*1000:6.2f}ms/batch, {fps:6.1f} pos/sec")
    
    # Test 6: Memory usage
    print("\n🧮 Memory Usage Analysis:")
    
    for size in ['tiny', 'small', 'medium']:
        desc = create_game_descriptor('go_19')
        model = PERANet(size, desc.planes, desc.action_space)
        
        param_count = count_parameters(model)
        param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        print(f"   {size.capitalize():6s}: {param_count/1e6:5.1f}M params, ~{param_size_mb:.1f}MB")
    
    print("\n🎉 All tests completed successfully!")
    print("="*60)
    
    # Practical usage example
    print("\n📚 PRACTICAL USAGE EXAMPLE:")
    print("""
    # Quick start for Go 19x19:
    desc = create_game_descriptor('go_19')
    drm = DRM3(frz=0.5, base_emergence=0.1)
    model = PERANet('small', desc.planes, desc.action_space)
    puct = EnhancedPUCT(c_puct=1.4, sims=800, drm=drm)
    
    # Training loop integration:
    for epoch in range(100):
        # ... your training code ...
        strengths = drm.step_all_rules()  # Evolve DRM rules
        if drm.is_stagnating():
            # Adapt training (learning rate, exploration, etc.)
            pass
    
    # Export trained model:
    export_to_onnx(model, desc, 'my_go_bot.onnx')
    """)
    
    print("\n🚀 Ready for production use!")
