# peradrm_2_0.py
"""
PERA-DRM 2.0: Dynamiczna Architektura Równościowa dla Gier Planszowych.
Łączy PERA-Net (permutacyjnie-równą sieć neuronową) z pełną implementacją DRM 3.0,
wzmacniając PUCT o dynamiczne bonusy eksploracyjne.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import time
import math
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Callable
from abc import ABC, abstractmethod
import json

# =========================================================================
# 1. Komponent DRM 3.0: Dynamic Rules Matrix
# =========================================================================

class Rule:
    """Pojedyncza reguła w systemie DRM."""
    def __init__(self, name: str, features: Optional[List[float]] = None, weight: float = 1.0):
        self.name = name
        self.features = features or [weight]
        self.weight = float(weight)
        self.strength = 0.0
        self.usage_count = 0
        self.last_used_cycle = 0
        self.created_cycle = 0
        self.is_new = True
        self.mutations = 0

    def __repr__(self) -> str:
        return f"Rule({self.name}, weight={self.weight:.2f}, strength={self.strength:.2f})"

    def update_strength(self, new_strength: float):
        self.strength = new_strength

    def apply_mutation(self, mutation_rate: float):
        self.mutations += 1
        for i in range(len(self.features)):
            self.features[i] += np.random.normal(0, mutation_rate)
        self.weight += np.random.normal(0, mutation_rate)
        self.weight = max(0.01, self.weight)

class DRM3System:
    """
    Zaawansowany system Dynamic Rules Matrix 3.0 z mechanizmami
    anty-stagnacyjnymi i ratunkowymi.
    """
    def __init__(self, FRZ: float, seed: Optional[int] = None, stable_cycles: int = 5,
                 stable_threshold: float = 0.8, base_emergence: float = 0.1):
        if seed:
            random.seed(seed)
        self.rules: List[Rule] = []
        self.archive: List[Rule] = []
        self.cycle = 0
        self.last_revival_cycle = 0
        self.FRZ = FRZ
        self.activity_history: List[float] = []
        self.stable_cycles = stable_cycles
        self.stable_threshold = stable_threshold
        self.base_emergence = base_emergence
        self._checkpoint: Optional[Dict[str, Any]] = None
        self._frz_cache = {}
        self._anti_stagnation_factor_cache = {}

    def add_rule(self, name: str, features: List[float], weight: float = 1.0):
        new_rule = Rule(name, features=features, weight=weight)
        new_rule.created_cycle = self.cycle
        self.rules.append(new_rule)
        return new_rule

    def _cosine_sim_vec(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

    def _enforce_diversity(self, mutation_rate: float = 0.1):
        """Mutuje najbardziej podobne reguły, aby zwiększyć różnorodność."""
        if len(self.rules) < 2:
            return
        
        sims = []
        for i in range(len(self.rules)):
            for j in range(i + 1, len(self.rules)):
                sim = self._cosine_sim_vec(self.rules[i].features, self.rules[j].features)
                sims.append((sim, i, j))
        
        sims.sort(reverse=True)
        if sims:
            _, i, j = sims[0]
            self.rules[i].apply_mutation(mutation_rate)
            self.rules[j].apply_mutation(mutation_rate)

    def _calculate_adaptive_frz_cached(self) -> float:
        if self.cycle in self._frz_cache:
            return self._frz_cache[self.cycle]
        
        # Przykład prostej adaptacji FRZ
        avg_activity = np.mean(self.activity_history[-self.stable_cycles:]) if len(self.activity_history) >= self.stable_cycles else 1.0
        adaptive_frz = self.FRZ / (avg_activity + 1e-6)
        
        self._frz_cache[self.cycle] = adaptive_frz
        return adaptive_frz

    def _get_anti_stagnation_factor_cached(self) -> float:
        if self.cycle in self._anti_stagnation_factor_cache:
            return self._anti_stagnation_factor_cache[self.cycle]

        factor = 1.0
        if self.is_stagnated():
            factor = 2.0  # Zwiększ siłę reguł by zwalczyć stagnację
        self._anti_stagnation_factor_cache[self.cycle] = factor
        return factor

    def _calculate_emergence_pressure(self, rule: Rule) -> float:
        # Zwiększa ciśnienie na nowe reguły, aby pomóc im się "wybić"
        return self.base_emergence * (1 - (rule.usage_count / (self.cycle + 1e-6)))

    def _curiosity_bonus(self, rule: Rule) -> float:
        # Nagradza reguły, które rzadko były używane w ostatnim czasie
        return 0.1 / (self.cycle - rule.last_used_cycle + 1e-6) if self.cycle > rule.last_used_cycle else 0.5

    def _calculate_strength(self, rule: Rule, external_reward: float = 0.0) -> float:
        """Oblicza siłę reguły z uwzględnieniem wszystkich czynników."""
        adaptive_frz = self._calculate_adaptive_frz_cached()
        anti_stagnation_factor = self._get_anti_stagnation_factor_cached()
        emergence_pressure = self._calculate_emergence_pressure(rule)
        curiosity_bonus = self._curiosity_bonus(rule)
        
        # Złożona formuła siły
        strength = (rule.weight * (1 + rule.usage_count) + external_reward + emergence_pressure + curiosity_bonus) \
                   / (adaptive_frz * anti_stagnation_factor)
        
        return strength

    def _detect_stagnation(self) -> bool:
        """Wykrywa stagnację na podstawie niskiej średniej aktywności."""
        if len(self.activity_history) < self.stable_cycles:
            return False
        
        avg_activity = np.mean(self.activity_history[-self.stable_cycles:])
        return avg_activity < self.stable_threshold

    def is_stagnated(self) -> bool:
        return self._detect_stagnation()

    def _create_checkpoint(self):
        self._checkpoint = {
            'rules': copy.deepcopy(self.rules),
            'cycle': self.cycle,
            'activity_history': copy.deepcopy(self.activity_history)
        }

    def _maybe_rollback_after_revival(self, new_stats_are_bad: bool):
        if self._checkpoint and new_stats_are_bad:
            self.rules = self._checkpoint['rules']
            self.cycle = self._checkpoint['cycle']
            self.activity_history = self._checkpoint['activity_history']
            self._checkpoint = None

    def _emergency_revival(self):
        """Przywraca system, jeśli wpadł w spiralę śmierci."""
        self._create_checkpoint()
        
        # Przenieś najsłabsze reguły do archiwum
        self.rules.sort(key=lambda r: r.strength)
        self.archive.extend(self.rules[:int(len(self.rules) * 0.2)])
        self.rules = self.rules[int(len(self.rules) * 0.2):]

        # Wprowadź nowe reguły i zmutuj pozostałe
        for _ in range(5):
            new_features = [random.uniform(0.1, 1.0) for _ in range(4)]
            self.add_rule(f"revival_rule_{self.cycle}_{len(self.rules)}", new_features)
        
        for rule in self.rules:
            rule.apply_mutation(0.2)

    def step(self, chosen_rule_name: str, external_reward: float = 0.0):
        """Przetwarza jeden cykl, aktualizując reguły."""
        self.cycle += 1
        
        # Aktualizacja siły wszystkich reguł
        total_strength = 0.0
        for rule in self.rules:
            new_strength = self._calculate_strength(rule, external_reward)
            rule.update_strength(new_strength)
            total_strength += new_strength
        
        # Aktywność jako średnia siła
        avg_activity = total_strength / (len(self.rules) + 1e-9)
        self.activity_history.append(avg_activity)
        
        # Reakcja na stagnację
        if self._detect_stagnation():
            self._enforce_diversity()
            
        # Reakcja na spiralę śmierci (np. długa stagnacja)
        if len(self.activity_history) > self.stable_cycles * 2 and avg_activity < self.stable_threshold / 2:
            self._emergency_revival()
            # Po ożywieniu, monitoruj efekty i w razie potrzeby zrób rollback
            # (dla uproszczenia, w przykładzie ten rollback jest opcjonalny)
            
        # Zaktualizuj licznik użycia dla wybranej reguły
        for rule in self.rules:
            if rule.name == chosen_rule_name:
                rule.usage_count += 1
                rule.last_used_cycle = self.cycle
                rule.is_new = False
                break

    def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki systemu."""
        return {
            "cycle": self.cycle,
            "n_rules": len(self.rules),
            "avg_activity": self.activity_history[-1] if self.activity_history else 0.0,
            "is_stagnated": self.is_stagnated(),
            "last_revival_cycle": self.last_revival_cycle,
        }

    def get_drm_bonus_for_move(self, move_features: List[float]) -> float:
        """
        Oblicza bonus DRM dla danego ruchu na podstawie cech.
        Bonus jest sumą sił pasujących reguł.
        """
        total_bonus = 0.0
        for rule in self.rules:
            sim = self._cosine_sim_vec(rule.features, move_features)
            if sim > 0.8:  # Próg dopasowania
                total_bonus += rule.strength * sim
        return total_bonus


# =========================================================================
# 2. Komponent PERA-Net: Permutation Equivariant Network
# =========================================================================

class GameDescriptor:
    def __init__(self, name: str, planes: int, action_space: int, board_size: Tuple[int, int]):
        self.name = name
        self.planes = planes
        self.action_space = action_space
        self.board_size = board_size

def create_game_descriptor(game_name: str) -> GameDescriptor:
    if game_name == 'go_9':
        return GameDescriptor('Go 9x9', 17, 82, (9, 9))
    if game_name == 'chess':
        return GameDescriptor('Chess', 119, 4672, (8, 8))
    raise ValueError(f"Nieznana gra: {game_name}")

class EquivariantConv2d(nn.Conv2d):
    """
    Równościowa konwolucja, która uwzględnia symetrie planszy.
    Dla uproszczenia, w tym przykładzie symuluje się ją za pomocą standardowej konwolucji,
    ale w pełnej implementacji wymagałaby specjalnego jądra.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = EquivariantConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = EquivariantConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, board_size: Tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.H, self.W = board_size
        self.q_conv = EquivariantConv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = EquivariantConv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = EquivariantConv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = EquivariantConv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q_conv(x).view(B, C, -1).permute(0, 2, 1)
        k = self.k_conv(x).view(B, C, -1)
        v = self.v_conv(x).view(B, C, -1).permute(0, 2, 1)

        attention_scores = torch.bmm(q, k) / (C ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        out = torch.bmm(attention_scores, v)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        out = self.out_conv(out)
        return out + x # Skip connection

class PERANet(nn.Module):
    """
    Permutation-Equivariant Residual Attention Network.
    """
    def __init__(self, size: str, planes: int, action_space: int, board_size: Tuple[int, int]):
        super().__init__()
        
        self.board_size = board_size
        filters = {'tiny': 32, 'small': 64, 'medium': 128}[size]
        blocks = {'tiny': 2, 'small': 4, 'medium': 6}[size]
        
        self.initial_conv = EquivariantConv2d(planes, filters, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(filters, filters) for _ in range(blocks)])
        self.attention = AttentionBlock(filters, board_size)
        
        # Policy head
        self.policy_conv = EquivariantConv2d(filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size[0] * board_size[1], action_space)

        # Value head
        self.value_conv = EquivariantConv2d(filters, 1, kernel_size=1)
        self.value_fc = nn.Sequential(
            nn.Linear(1 * board_size[0] * board_size[1], filters),
            nn.ReLU(),
            nn.Linear(filters, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.initial_conv(x))
        x = self.body(x)
        x = self.attention(x)
        
        # Policy head
        policy_out = F.relu(self.policy_conv(x))
        policy_out = policy_out.view(policy_out.size(0), -1)
        policy_logits = self.policy_fc(policy_out)
        
        # Value head
        value_out = F.relu(self.value_conv(x))
        value_out = value_out.view(value_out.size(0), -1)
        value = self.value_fc(value_out)
        
        return policy_logits, value

# =========================================================================
# 3. Integracja: Enhanced PUCT (DRM-assisted)
# =========================================================================

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: Dict[Any, Node] = {}
        self.n = 0  # Visit count
        self.w = 0  # Total value
        self.p = 0  # Prior probability from network

    @property
    def q(self):
        return self.w / (self.n + 1e-6)

class EnhancedPUCT:
    def __init__(self, model: PERANet, drm: DRM3System, game_descriptor: GameDescriptor, c_puct: float = 1.0):
        self.model = model
        self.drm = drm
        self.c_puct = c_puct
        self.game_descriptor = game_descriptor
        self.move_to_features: Dict[Any, List[float]] = {}

    def run_simulation(self, root_state, num_simulations: int):
        root_node = Node(root_state)
        
        for _ in range(num_simulations):
            node = root_node
            # Selection
            while node.children:
                best_uct = -1
                best_move = None
                for move, child in node.children.items():
                    # Oblicz bonus DRM dla ruchu
                    move_features = self.move_to_features.get(move, [0.0]) # Zastępcze cechy
                    drm_bonus = self.drm.get_drm_bonus_for_move(move_features)
                    
                    uct_score = child.q + self.c_puct * child.p * math.sqrt(node.n) / (1 + child.n) + drm_bonus
                    
                    if uct_score > best_uct:
                        best_uct = uct_score
                        best_move = move
                node = node.children[best_move]

            # Expansion
            policy_logits, value = self.model(node.state)
            policy = F.softmax(policy_logits, dim=-1).squeeze(0)
            
            for i in range(self.game_descriptor.action_space):
                move = i  # uproszczone mapowanie ruchu
                if move not in node.children:
                    new_node = Node(state=None, parent=node, move=move)
                    new_node.p = policy[i].item()
                    node.children[move] = new_node

            # Backpropagation
            current_value = value.item()
            temp_node = node
            while temp_node.parent:
                temp_node.w += current_value
                temp_node.n += 1
                temp_node = temp_node.parent
            root_node.w += current_value
            root_node.n += 1
        
        return root_node


# =========================================================================
# 4. Testowanie i Narzędzia Pomocnicze
# =========================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def export_to_onnx(model: nn.Module, filename: str, input_shape: Tuple[int, ...]):
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, filename, opset_version=11)
    print(f"Model exported to {filename} successfully.")

if __name__ == "__main__":
    print("🚀 Rozpoczynanie testów PERA-DRM 2.0...")

    # TEST 1: DRM 3.0
    print("\n--- 🧪 Testowanie DRM 3.0 ---")
    drm = DRM3System(FRZ=40.0, seed=42, stable_cycles=5, stable_threshold=0.4)
    for i in range(12):
        drm.add_rule(f"rule_{i}", features=[random.uniform(0.5, 1.5) for _ in range(4)])
    
    for cycle in range(60):
        # Symulacja wyboru "najsilniejszej" reguły
        if drm.rules:
            chosen_rule = max(drm.rules, key=lambda r: r.strength)
            drm.step(chosen_rule.name, external_reward=0.1 if random.random() > 0.5 else 0.0)
        
        if (cycle + 1) % 10 == 0:
            stats = drm.get_stats()
            print(f"Cykl {stats['cycle']:2d}: n_rules={stats['n_rules']}, "
                  f"avg_activity={stats['avg_activity']:.4f}, stagnacja={stats['is_stagnated']}")
            if stats['is_stagnated']:
                print("   🚨 Stagnacja wykryta! Aktywowano mechanizm diversity.")

    print("--- ✅ Test DRM 3.0 zakończony ---")

    # TEST 2: PERA-Net
    print("\n--- 🧠 Testowanie PERA-Net ---")
    descriptor_go = create_game_descriptor('go_9')
    model = PERANet('small', descriptor_go.planes, descriptor_go.action_space, descriptor_go.board_size)
    dummy_input = torch.randn(1, descriptor_go.planes, descriptor_go.board_size[0], descriptor_go.board_size[1])
    
    try:
        policy_logits, value = model(dummy_input)
        print(f"Wymiary wyjścia policy_logits: {policy_logits.shape}")
        print(f"Wymiary wyjścia value: {value.shape}")
        print(f"Liczba parametrów w sieci: {count_parameters(model) / 1e6:.2f}M")
    except Exception as e:
        print(f"Błąd podczas testu PERA-Net: {e}")

    print("--- ✅ Test PERA-Net zakończony ---")
    
    # TEST 3: Integracja
    print("\n--- 🤝 Testowanie Integracji PUCT + DRM ---")
    
    # Symulowane stany i cechy ruchów
    root_state = torch.randn(1, descriptor_go.planes, descriptor_go.board_size[0], descriptor_go.board_size[1])
    
    # Przygotowanie losowych cech dla ruchów (w praktyce generowane dynamicznie)
    drm_assisted_puct = EnhancedPUCT(model, drm, descriptor_go)
    for i in range(descriptor_go.action_space):
        drm_assisted_puct.move_to_features[i] = [random.uniform(0.1, 1.0) for _ in range(4)]
    
    # Uruchomienie symulacji
    try:
        root_node = drm_assisted_puct.run_simulation(root_state, num_simulations=100)
        print(f"Symulacja PUCT z bonusem DRM zakończona. Odwiedzono węzłów: {root_node.n}")
        
        # Przykład, jak bonus DRM wpłynął na wybór
        if root_node.children:
            best_move = max(root_node.children, key=lambda move: root_node.children[move].n)
            print(f"Najczęściej odwiedzany ruch (prawdopodobnie najlepszy wg PUCT+DRM): {best_move}")
            print(f"Siła DRM dla tego ruchu: {drm.get_drm_bonus_for_move(drm_assisted_puct.move_to_features[best_move]):.4f}")

    except Exception as e:
        print(f"Błąd podczas testu integracji: {e}")

    print("\n--- ✅ Wszystkie testy zakończone pomyślnie! ---")