# A 56 Gb/s triangle from three $10 cards

Splitting a model across boxes needs a fast link between them, and fast links are usually where a homelab budget
ends. Ours cost **about $136 for three boxes**: three used Mellanox ConnectX-3 cards at $10.19 each and a lot of five 2 m
copper cables for $105.19 with shipping. Each pair of boxes gets its own cable, so there is no switch. Each link moves
**52.3 to 52.7 Gb/s** of RDMA traffic and about 50 Gb/s of parallel TCP.

Measured 2026-10-23 and 2026-10-24 (UTC). The model-split work these links were bought for has not run yet: see the
last section.

## Box

| box | slot the card is in | other network |
|---|---|---|
| **lassie** (2017 server, box D) | PCIe 3.0 x8 | 1 GbE |
| **droopy** (2019 workstation, box B) | PCIe 3.0 x16 | 10 GbE |
| **odie** (Threadripper, box C) | x8, beside the two B70s | 10 GbE |

A PCIe 3.0 x8 slot moves about 63 Gb/s after line encoding, and less once the bus's own protocol overhead is paid, so
the narrowest slot here leaves a faster card little headroom over the 52-53 Gb/s these cards measured (our estimate;
we have not tried a faster card in these slots). That is why we bought the $10 card. The fourth box (snoopy) stays on
10 GbE.

```
odie p1 ===== link C ===== droopy p2
odie p2 ===== link B ===== lassie p1
lassie p2 === link A ===== droopy p1
```

## At a glance

| | |
|---|---|
| cards | 3x Mellanox MCX354A-QCBT (ConnectX-3, dual-port QSFP, board rev A9), used, $10.19 each |
| cables | QSFP28 100G passive copper DAC, 2 m, lot of 5 for $99 + $6.19 shipping (rated far above what these cards need) |
| firmware | each card cross-flashed to the FCBT image, FW 2.42.5000, which unlocks 56 Gb/s (FDR InfiniBand, 56GbE) |
| result | RDMA (RoCE, 1 MiB writes, 4 queue pairs): **52.56 / 52.36 / 52.30 Gb/s** on links A / B / C; TCP, 8 streams: about 50 Gb/s |
| mode now | 56GbE Ethernet on all six ports, MTU 9000 |

## Details

### The cheap card is the fast card with other firmware

The QCBT and the FCBT are the same board. Only the firmware's board ID (PSID) differs, and with it the top speed: the
QCBT firmware stops at 40 Gb/s (QDR), the FCBT firmware runs 56 Gb/s (FDR). Used QCBTs sold for about $11 against $40 and
up for an FCBT. Our first card linked at 40 (QDR) out of the box, `ib_write_bw` 29.3 Gb/s on a loop from one port to the
other. After the flash and a full power cycle it linked at 56 (FDR).

The FCBT firmware image is named for board revisions A2 to A5, and our cards are rev A9. It worked on all three.

### Every number, per link

All runs card to card, after the flash, with perftest and iperf3.

**InfiniBand mode** (first test, 4X FDR, rate 56 on all six ports):

| link | `ib_write_bw`, 1 MiB, 4 QPs | `ib_write_lat`, 2 B, typical |
|---|---|---|
| A | 52.67 Gb/s | 0.96 us |
| B | 52.63 Gb/s | 1.00 us |
| C | 52.60 Gb/s | 0.94 us |

FDR's data rate is about 54.5 Gb/s, so RDMA runs at about 97 % of the wire. TCP over IPoIB (connected mode, MTU 65520)
reached only 19.0 Gb/s with one stream and 16.4 with four: on ConnectX-3, IPoIB is bound by the CPU.

**56GbE Ethernet mode** (what runs now):

| link | iperf3, 1 stream | iperf3, 8 streams | RoCE `ib_write_bw -R`, 1 MiB, 4 QPs |
|---|---|---|---|
| A | 19.9 Gb/s | 50.4 Gb/s | 52.56 Gb/s |
| B | 22.6 Gb/s | 50.2 Gb/s | 52.36 Gb/s |
| C | 22.3 Gb/s | 15.7 Gb/s; 48.1 at 4 streams | 52.30 Gb/s |

Ethernet keeps RDMA at the same speed (as RoCE) and gives parallel TCP about 2.6x the best IPoIB figure (50.4 Gb/s
against 19.0). Software that talks
plain TCP, like llama.cpp's RPC backend, sees the TCP column; software that uses RDMA sees the last. Link C's 8-stream
run came out low once (15.7); a rerun at 4 streams gave 48.1, and we have not chased the outlier.

## Reproduce

Flash, per card (`mstflint` from the `mstflint` package):

```bash
lspci -d 15b3:                                  # find the card, e.g. 81:00.0
sudo mstflint -d 81:00.0 q                      # QCBT reads PSID MT_1090110018; note the firmware version
sudo mstflint -d 81:00.0 ri cx3-orig.bin        # back up the whole current image first
sudo mstflint -d 81:00.0 -i fw-ConnectX3-rel-2_42_5000-MCX354A-FCB_A2-A5-FlexBoot-3.4.752.bin \
  -allow_psid_change burn                       # keeps the card's own GUIDs and MACs
# power the box off and on again (a warm reboot is not enough), then:
sudo mstflint -d 81:00.0 q                      # expect PSID MT_1090120019, FW 2.42.5000
```

The firmware comes from NVIDIA's Mellanox firmware downloads page. Check its checksum against the one published there
before you burn it. A backup image burns back the same way, and a card that stops answering can be put into
flash-recovery mode with its recovery jumper.

Ethernet mode at 56 Gb/s, per box:

```bash
echo 'options mlx4_core port_type_array=2,2' | sudo tee /etc/modprobe.d/mlx4-eth.conf   # both ports Ethernet at boot
sudo ethtool -s <netdev> speed 56000 autoneg off   # on both ends; autonegotiation settles at 40GbE
```

We made the speed setting persistent with a udev rule that runs the same `ethtool` line on every mlx4 network device,
and gave each link its own small subnet with MTU 9000. Run RoCE tests with perftest's `-R` (connect through rdma_cm).

## What did not work

- **Flashing a box with Secure Boot on.** The kernel's lockdown mode blocks `mstflint`'s PCI access on that box.
- **The distribution's `mstflint` on Ubuntu 26.04** (4.33) crashed on `q` on another box.
- What fixed both: **flash over the cable from a box where `mstflint` works.** With the cards in InfiniBand mode and
  the subnet manager running, `mstflint` (4.26 here) reaches a remote card by its link address: `mstflint -d lid-2 q`
  for the card at the other end of the first port, `-d lid-1,<rdma-device>,2` through the second port. The same backup
  and burn commands work this way.
- **perftest's `-x` GID index hung** on these cards in Ethernet mode; `-R` works.
- **Running perftest's 4-QP test as a normal user** failed with "Couldn't create CQ" on one box (the locked-memory
  limit); it ran as root.

## Not measured yet

- The point of the cards: a model split across two boxes at 56G against 10G, and a shared KV cache in one box's RAM
  for the others. Neither ran before the research paused (2026-10-26); the links are measured and ready for it.
- Whether every setting comes back by itself after each box's next reboot.
- Wall-plug power of the cards.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
