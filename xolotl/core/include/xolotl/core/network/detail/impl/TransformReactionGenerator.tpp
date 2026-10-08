#pragma once

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
template <typename TBase>
TransformReactionGenerator<TBase>::TransformReactionGenerator(
	const NetworkType& network) :
	Superclass(network),
	_clusterTransformReactionCounts(
		"Transform Reaction Counts", Superclass::getNumberOfClusters() + 3)
{
}

template <typename TBase>
typename TransformReactionGenerator<TBase>::IndexType
TransformReactionGenerator<TBase>::getRowMapAndTotalReactionCount()
{
	_numPrecedingReactions = Superclass::getRowMapAndTotalReactionCount();
	_numTransformReactions = Kokkos::get_crs_row_map_from_counts(
		_transformCrsRowMap, _clusterTransformReactionCounts);

	_transformReactions = Kokkos::View<TransformReactionType*>(
		"Transform Reactions", _numTransformReactions);

	return _numPrecedingReactions + _numTransformReactions;
}

template <typename TBase>
void
TransformReactionGenerator<TBase>::setupCrsClusterSetSubView()
{
	Superclass::setupCrsClusterSetSubView();
	_transformCrsClusterSets =
		this->getClusterSetSubView(std::make_pair(_numPrecedingReactions,
			_numPrecedingReactions + _numTransformReactions));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
TransformReactionGenerator<TBase>::addTransformReaction(
	Count, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableStdReaction())
		return;
	// Reactions of the extra DOFs all go in the last bucket
	auto bucket = (clusterSet.cluster0 < this->_clusterData.numClusters) ?
		clusterSet.cluster0 :
		this->_clusterData.numClusters;
	Kokkos::atomic_inc(&_clusterTransformReactionCounts(bucket));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
TransformReactionGenerator<TBase>::addTransformReaction(
	Construct, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableStdReaction())
		return;

	auto bucket = (clusterSet.cluster0 < this->_clusterData.numClusters) ?
		clusterSet.cluster0 :
		this->_clusterData.numClusters;
	auto id = _transformCrsRowMap(bucket);
	for (; !util::atomicCompareExchangeStrong(
			 &_transformCrsClusterSets(id).cluster0,
			 NetworkType::invalidIndex(), clusterSet.cluster0);
		++id) { }
	_transformCrsClusterSets(id) = clusterSet;
}
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
