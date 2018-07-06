
set(SOURCE_FILES ${SOURCE_FILES}
        block_addressing_logic.hpp
        blocked_bloomfilter.hpp
        blocked_bloomfilter_batch_dispatch.hpp
        blocked_bloomfilter_block_logic.hpp
        blocked_bloomfilter_block_logic_sgew.hpp
        blocked_bloomfilter_block_logic_sltw.hpp
        blocked_bloomfilter_block_logic_zoned.hpp
        blocked_bloomfilter_config.hpp
        blocked_bloomfilter_logic.hpp
        blocked_bloomfilter_tune.hpp
        blocked_bloomfilter_tune_impl.hpp

        fpr.hpp
        hash_family.hpp
        math.hpp
        vector_helper.hpp

        zoned_blocked_bloomfilter.hpp
        zoned_blocked_bloomfilter_tune_impl.hpp

    PARENT_SCOPE)
